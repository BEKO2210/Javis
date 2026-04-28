//! Javis live visualisation server (binary).
//!
//! Serves the static frontend and exposes `/ws?query=<word>`. All
//! request-handling and pipeline-streaming logic lives in
//! `viz::server` so the same code is exercised by integration tests.

use std::net::SocketAddr;
use std::path::PathBuf;

use viz::server::router;

#[tokio::main]
async fn main() {
    let static_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
    let app = router(static_dir);

    let addr: SocketAddr = "127.0.0.1:7777".parse().unwrap();
    println!("javis-viz listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
