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
//!
//! ## Logging
//!
//! Logs are emitted via the `tracing` crate. The verbosity is
//! controlled by `RUST_LOG` (e.g. `RUST_LOG=info`, `RUST_LOG=viz=debug`)
//! and defaults to `info`. Set `JAVIS_LOG_FORMAT=json` to switch from
//! human-readable to JSON-structured output for log aggregation.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use viz::server::router;
use viz::AppState;

#[tokio::main]
async fn main() {
    init_tracing();
    viz::metrics::init();

    let static_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");

    let snapshot_path = parse_snapshot_arg();
    let state = Arc::new(AppState::new());

    if let Some(path) = &snapshot_path {
        if path.exists() {
            match state.load_from_file(path).await {
                Ok(()) => {
                    let (sentences, words) = state.stats().await;
                    info!(
                        path = %path.display(),
                        sentences,
                        words,
                        "loaded snapshot",
                    );
                }
                Err(e) => {
                    error!(
                        path = %path.display(),
                        error = %e,
                        "failed to load snapshot; bootstrapping fresh",
                    );
                    state.bootstrap_default_corpus(None).await;
                }
            }
        } else {
            info!(
                path = %path.display(),
                "no snapshot found; bootstrapping default corpus",
            );
            state.bootstrap_default_corpus(None).await;
        }
    } else {
        info!("bootstrapping brain on default corpus");
        state.bootstrap_default_corpus(None).await;
    }

    let (sentences, words) = state.stats().await;
    info!(sentences, words, "brain ready");

    let app = router(state.clone(), static_dir);
    let addr: SocketAddr = "127.0.0.1:7777".parse().unwrap();
    info!(%addr, "javis-viz listening");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());
    if let Err(e) = server.await {
        error!(error = %e, "server error");
    }

    if let Some(path) = snapshot_path {
        match state.save_to_file(&path).await {
            Ok(()) => info!(path = %path.display(), "snapshot written"),
            Err(e) => error!(
                path = %path.display(),
                error = %e,
                "failed to write snapshot",
            ),
        }
    }
}

/// Initialise the `tracing` subscriber.
///
/// Defaults to `info`-level human-readable output. Override with
/// `RUST_LOG` (e.g. `RUST_LOG=viz=debug`) for finer control. Set
/// `JAVIS_LOG_FORMAT=json` to emit JSON-structured logs (useful for
/// production log aggregation).
fn init_tracing() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,viz=info"));

    let json_format = std::env::var("JAVIS_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    let registry = tracing_subscriber::registry().with(env_filter);
    if json_format {
        registry.with(fmt::layer().json()).init();
    } else {
        registry.with(fmt::layer().with_target(false)).init();
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
    info!("shutdown signal received");
}
