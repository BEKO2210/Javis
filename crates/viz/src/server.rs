//! Axum router + WebSocket session, factored out so both the binary
//! (`src/main.rs`) and integration tests can share the same wiring.
//!
//! The server holds one persistent [`AppState`]. Each WebSocket
//! request triggers either a training pass (`?action=train&text=…`),
//! a recall (`?action=recall&query=…`) or a state reset
//! (`?action=reset`). All event traffic streams over the same JSON
//! schema described in [`crate::events`].

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use tokio::sync::mpsc;

use crate::events::Event;
use crate::state::AppState;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Action {
    #[default]
    Recall,
    Train,
    Reset,
    Ask,
}

#[derive(Debug, Default, Deserialize)]
pub struct WsParams {
    #[serde(default)]
    action: Action,
    /// For `recall` and `ask` — the query keyword / question.
    #[serde(default)]
    query: Option<String>,
    /// For `train` — the sentence to learn.
    #[serde(default)]
    text: Option<String>,
    /// For `ask` — the full RAG context payload.
    #[serde(default)]
    rag: Option<String>,
    /// For `ask` — the compact Javis context payload.
    #[serde(default)]
    javis: Option<String>,
}

/// Build the full router (static-file fallback + `/ws` endpoint).
pub fn router(state: Arc<AppState>, static_dir: PathBuf) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state)
        .fallback_service(tower_http::services::ServeDir::new(static_dir))
}

/// Bare router without static-file serving — handy for tests.
pub fn router_no_static(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Query(params): Query<WsParams>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| run_session(socket, state, params))
}

pub async fn run_session(mut socket: WebSocket, state: Arc<AppState>, params: WsParams) {
    let (tx, mut rx) = mpsc::channel::<Event>(1024);

    let handle_state = state.clone();
    let handle = tokio::task::spawn(async move {
        match params.action {
            Action::Recall => {
                let query = params.query.unwrap_or_else(|| "rust".to_string());
                handle_state.run_recall(query, tx).await;
            }
            Action::Train => {
                let sentence = params.text.unwrap_or_default();
                if sentence.trim().is_empty() {
                    let _ = tx
                        .send(Event::Phase {
                            name: "error".into(),
                            detail: "empty training text".into(),
                        })
                        .await;
                    let _ = tx.send(Event::Done).await;
                    return;
                }
                handle_state.run_train(sentence, Some(tx)).await;
            }
            Action::Reset => {
                handle_state.reset().await;
                let _ = tx
                    .send(Event::Phase {
                        name: "reset".into(),
                        detail: "brain wiped".into(),
                    })
                    .await;
                let _ = tx.send(Event::Done).await;
            }
            Action::Ask => {
                let question = params.query.unwrap_or_default();
                let rag = params.rag.unwrap_or_default();
                let javis = params.javis.unwrap_or_default();
                if question.trim().is_empty() {
                    let _ = tx
                        .send(Event::Phase {
                            name: "error".into(),
                            detail: "ask requires a query".into(),
                        })
                        .await;
                    let _ = tx.send(Event::Done).await;
                    return;
                }
                handle_state.run_ask(question, rag, javis, tx).await;
            }
        }
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
