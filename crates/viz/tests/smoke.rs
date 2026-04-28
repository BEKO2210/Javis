//! Smoke test: spawn the server in-process, hit `/ws?query=rust`, walk
//! the event stream until we see a `decoded` event, then `done`. Asserts
//! the contract enough that broken wiring fails on `cargo test`.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_init_phase_decoded_done() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = viz::server::router_no_static();
    let server = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let url = format!("ws://{addr}/ws?query=rust");
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();

    let mut saw_init = false;
    let mut saw_step = false;
    let mut saw_decoded = false;
    let mut saw_done = false;
    let mut decoded_payload: Option<serde_json::Value> = None;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while let Ok(Some(msg)) = tokio::time::timeout_at(deadline, ws.next()).await {
        let Ok(msg) = msg else { break };
        let Message::Text(text) = msg else { continue };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        match v.get("type").and_then(|t| t.as_str()) {
            Some("init") => saw_init = true,
            Some("step") => saw_step = true,
            Some("decoded") => {
                saw_decoded = true;
                decoded_payload = Some(v);
            }
            Some("done") => {
                saw_done = true;
                break;
            }
            _ => {}
        }
    }
    let _ = ws.send(Message::Close(None)).await;

    assert!(saw_init, "no init event");
    assert!(saw_step, "no step event");
    assert!(saw_decoded, "no decoded event");
    assert!(saw_done, "no done event");

    let d = decoded_payload.expect("decoded payload");
    let reduction = d
        .get("reduction_pct")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let rag_tokens = d.get("rag_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    assert!(rag_tokens > 0, "RAG returned empty payload");
    assert!(
        reduction >= 70.0,
        "expected ≥ 70 % reduction over the wire, got {reduction:.1}%",
    );

    server.abort();
}
