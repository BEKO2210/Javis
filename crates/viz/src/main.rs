//! Javis live visualisation server (binary).
//!
//! Holds one persistent brain for the lifetime of the process. The
//! default corpus is auto-trained at startup so the very first recall
//! already has something to surface; further training requests append
//! more sentences without re-initialising.
//!
//! ## Snapshots
//!
//! The brain can survive process restarts via JSON snapshots:
//!
//! ```sh
//! # load on startup, save on Ctrl-C
//! cargo run -p viz --release -- --snapshot brain.json
//! ```
//!
//! If the file exists, it's loaded and the bootstrap-corpus step is
//! skipped. On graceful shutdown (SIGINT) the current state is written
//! back to the same path.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use viz::server::router;
use viz::AppState;

#[tokio::main]
async fn main() {
    let static_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");

    let snapshot_path = parse_snapshot_arg();
    let state = Arc::new(AppState::new());

    if let Some(path) = &snapshot_path {
        if path.exists() {
            match state.load_from_file(path).await {
                Ok(()) => {
                    let (s, w) = state.stats().await;
                    println!(
                        "javis-viz: loaded snapshot from {} ({s} sentences, {w} concepts)",
                        path.display(),
                    );
                }
                Err(e) => {
                    eprintln!(
                        "javis-viz: failed to load snapshot {}: {e}; bootstrapping fresh",
                        path.display(),
                    );
                    state.bootstrap_default_corpus(None).await;
                }
            }
        } else {
            println!(
                "javis-viz: no snapshot at {}; bootstrapping default corpus",
                path.display(),
            );
            state.bootstrap_default_corpus(None).await;
        }
    } else {
        println!("javis-viz: bootstrapping brain on default corpus…");
        state.bootstrap_default_corpus(None).await;
    }

    let (sentences, words) = state.stats().await;
    println!("javis-viz: ready ({sentences} sentences, {words} concepts)",);

    let app = router(state.clone(), static_dir);
    let addr: SocketAddr = "127.0.0.1:7777".parse().unwrap();
    println!("javis-viz listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());
    if let Err(e) = server.await {
        eprintln!("javis-viz: server error: {e}");
    }

    if let Some(path) = snapshot_path {
        match state.save_to_file(&path).await {
            Ok(()) => println!("javis-viz: snapshot written to {}", path.display()),
            Err(e) => eprintln!(
                "javis-viz: failed to write snapshot {}: {e}",
                path.display(),
            ),
        }
    }
}

/// Parse a single optional `--snapshot <path>` argument. Anything more
/// elaborate would call for a CLI parser crate; this stays dependency-
/// free.
fn parse_snapshot_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--snapshot" => {
                return args.next().map(PathBuf::from);
            }
            other if other.starts_with("--snapshot=") => {
                return Some(PathBuf::from(&other[11..]));
            }
            _ => {}
        }
    }
    None
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    println!("javis-viz: shutdown signal received");
}
