//! Axum router + WebSocket session, factored out so both the binary
//! (`src/main.rs`) and integration tests can share the same wiring.

use std::path::PathBuf;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::Query;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use tokio::sync::mpsc;
use tower_http::services::ServeDir;

use crate::events::Event;
use crate::pipeline::run_demo_session;

#[derive(Debug, Deserialize)]
pub struct WsParams {
    #[serde(default = "default_query")]
    pub query: String,
}

fn default_query() -> String {
    "rust".to_string()
}

/// Build the full router, including the static-file fallback. Pass a
/// path-less router (no fallback) by calling `Router::new().route(...)`
/// directly if you don't want static files.
pub fn router(static_dir: PathBuf) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new(static_dir))
}

/// Bare router without static-file serving — handy for tests.
pub fn router_no_static() -> Router {
    Router::new().route("/ws", get(ws_handler))
}

async fn ws_handler(ws: WebSocketUpgrade, Query(params): Query<WsParams>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| run_session(socket, params.query))
}

pub async fn run_session(mut socket: WebSocket, query: String) {
    let (tx, mut rx) = mpsc::channel::<Event>(1024);
    let q = query.clone();
    let handle = tokio::task::spawn(async move {
        run_demo_session(q, tx).await;
    });

    while let Some(ev) = rx.recv().await {
        let payload = match serde_json::to_string(&ev) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if socket.send(Message::Text(payload)).await.is_err() {
            break;
        }
    }
    let _ = socket.close().await;
    let _ = handle.await;
}
