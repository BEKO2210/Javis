//! Javis live visualisation server (binary).
//!
//! Holds one persistent brain for the lifetime of the process. The
//! default corpus is auto-trained at startup so the very first recall
//! already has something to surface; further training requests append
//! more sentences without re-initialising.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use viz::server::router;
use viz::AppState;

#[tokio::main]
async fn main() {
    let static_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");

    let state = Arc::new(AppState::new());
    println!("javis-viz: bootstrapping brain on default corpus…");
    state.bootstrap_default_corpus(None).await;
    let (sentences, words) = state.stats().await;
    println!(
        "javis-viz: ready ({sentences} sentences, {words} concepts)",
    );

    let app = router(state, static_dir);
    let addr: SocketAddr = "127.0.0.1:7777".parse().unwrap();
    println!("javis-viz listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
