//! Axum router + WebSocket session, factored out so both the binary
//! (`src/main.rs`) and integration tests can share the same wiring.
//!
//! The server holds one persistent [`AppState`]. Each WebSocket
//! request triggers either a training pass (`?action=train&text=…`),
//! a recall (`?action=recall&query=…`) or a state reset
//! (`?action=reset`). All event traffic streams over the same JSON
//! schema described in [`crate::events`].

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, info, info_span, warn, Instrument};

use crate::events::Event;
use crate::metrics as viz_metrics;
use crate::state::AppState;

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Action {
    #[default]
    Recall,
    Train,
    Reset,
    Ask,
}

impl Action {
    fn as_str(self) -> &'static str {
        match self {
            Action::Recall => "recall",
            Action::Train => "train",
            Action::Reset => "reset",
            Action::Ask => "ask",
        }
    }
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

/// Build the full router (static-file fallback + `/ws` endpoint +
/// `/health` / `/ready` probes + `/metrics` Prometheus exposition).
pub fn router(state: Arc<AppState>, static_dir: PathBuf) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
        .fallback_service(tower_http::services::ServeDir::new(static_dir))
}

/// Bare router without static-file serving — handy for tests.
pub fn router_no_static(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

/// Liveness probe.
///
/// Returns `200 OK` as long as the HTTP runtime can answer at all.
/// Container orchestrators should hit this on a short interval; a
/// failure means restart the process.
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Readiness probe.
///
/// Returns `200 OK` plus a small JSON body describing the brain state
/// (sentence count, vocabulary size, LLM mode). Orchestrators should
/// only route traffic to the pod once this returns 200; before
/// `bootstrap_default_corpus` finishes, it can take a few hundred ms.
///
/// We don't fail this endpoint when the brain is "empty" — an empty
/// brain is a valid state right after `reset`. The response is
/// purely informational.
async fn ready_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (sentences, words) = state.stats().await;
    let body = ReadyBody {
        status: "ready",
        sentences,
        words,
        llm: if state.llm_is_real() { "real" } else { "mock" },
    };
    (StatusCode::OK, Json(body))
}

#[derive(Debug, Serialize)]
struct ReadyBody {
    status: &'static str,
    sentences: usize,
    words: usize,
    llm: &'static str,
}

/// Prometheus exposition endpoint.
///
/// Renders the current state of every counter / histogram / gauge in
/// the global recorder. If the recorder hasn't been installed (e.g.
/// during a unit test that never calls `metrics::init()`) the body
/// is empty but the status code is still 200 — Prometheus treats an
/// empty scrape as "no metrics yet", not a failure.
async fn metrics_handler() -> impl IntoResponse {
    let body = viz_metrics::render();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Query(params): Query<WsParams>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| run_session(socket, state, params))
}

/// Monotonic per-process WebSocket session counter. Embedded in every
/// log line via the `session` field so concurrent sessions can be
/// disambiguated in aggregated output.
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

pub async fn run_session(socket: WebSocket, state: Arc<AppState>, params: WsParams) {
    let session_id = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let span = info_span!(
        "ws_session",
        session = session_id,
        action = params.action.as_str(),
    );
    run_session_inner(socket, state, params)
        .instrument(span)
        .await;
}

async fn run_session_inner(mut socket: WebSocket, state: Arc<AppState>, params: WsParams) {
    info!("session started");
    metrics::counter!(
        "javis_ws_sessions_total",
        "action" => params.action.as_str(),
    )
    .increment(1);

    let (tx, mut rx) = mpsc::channel::<Event>(1024);

    let handle_state = state.clone();
    let handle = tokio::task::spawn(
        async move {
            match params.action {
                Action::Recall => {
                    let query = params.query.unwrap_or_else(|| "rust".to_string());
                    debug!(%query, "dispatching recall");
                    handle_state.run_recall(query, tx).await;
                }
                Action::Train => {
                    let sentence = params.text.unwrap_or_default();
                    if sentence.trim().is_empty() {
                        warn!("rejecting train: empty text");
                        let _ = tx
                            .send(Event::Phase {
                                name: "error".into(),
                                detail: "empty training text".into(),
                            })
                            .await;
                        let _ = tx.send(Event::Done).await;
                        return;
                    }
                    debug!(text_len = sentence.len(), "dispatching train");
                    handle_state.run_train(sentence, Some(tx)).await;
                }
                Action::Reset => {
                    info!("dispatching reset");
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
                        warn!("rejecting ask: empty query");
                        let _ = tx
                            .send(Event::Phase {
                                name: "error".into(),
                                detail: "ask requires a query".into(),
                            })
                            .await;
                        let _ = tx.send(Event::Done).await;
                        return;
                    }
                    debug!(
                        rag_len = rag.len(),
                        javis_len = javis.len(),
                        "dispatching ask",
                    );
                    handle_state.run_ask(question, rag, javis, tx).await;
                }
            }
        }
        .in_current_span(),
    );

    let mut events_sent: u64 = 0;
    while let Some(ev) = rx.recv().await {
        let payload = match serde_json::to_string(&ev) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if socket.send(Message::Text(payload)).await.is_err() {
            debug!(events_sent, "client closed socket early");
            break;
        }
        events_sent += 1;
    }
    let _ = socket.close().await;
    let _ = handle.await;
    info!(events_sent, "session ended");
}
