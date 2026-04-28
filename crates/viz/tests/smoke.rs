//! Smoke test: spawn the server in-process, train a sentence, recall a
//! word from it, assert the WebSocket stream conforms to the contract.

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn train_then_recall_streams_decoded() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(viz::AppState::new());
    let app = viz::server::router_no_static(state);
    let server = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Step 1: train one sentence so the brain has any vocabulary.
    let train_url = format!(
        "ws://{addr}/ws?action=train&text={}",
        urlencode("Rust is a systems language focused on memory safety and ownership."),
    );
    walk_until_done(&train_url).await;

    // Step 2: recall a word that's in the trained sentence. The
    // dictionary should know it now.
    let recall_url = format!("ws://{addr}/ws?action=recall&query=rust");
    let decoded = walk_until_done(&recall_url).await;

    let d = decoded.expect("no decoded event from recall");
    let candidates = d.get("candidates").and_then(|v| v.as_array()).unwrap();
    let words: Vec<String> = candidates
        .iter()
        .filter_map(|c| c.get("word").and_then(|w| w.as_str()).map(|s| s.to_string()))
        .collect();
    assert!(
        words.iter().any(|w| w == "rust"),
        "decoded candidates did not include 'rust': {words:?}",
    );
    let reduction = d.get("reduction_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
    assert!(
        reduction >= 70.0,
        "expected ≥ 70 % token reduction over the wire, got {reduction:.1}%",
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_clears_dictionary() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(viz::AppState::new());
    state
        .run_train("Hello world from javis.".into(), None)
        .await;
    let (sentences_before, words_before) = state.stats().await;
    assert!(sentences_before >= 1 && words_before >= 1);

    let app = viz::server::router_no_static(state.clone());
    let server = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let reset_url = format!("ws://{addr}/ws?action=reset");
    walk_until_done(&reset_url).await;

    let (sentences_after, words_after) = state.stats().await;
    assert_eq!(sentences_after, 0);
    assert_eq!(words_after, 0);

    server.abort();
}

async fn walk_until_done(url: &str) -> Option<serde_json::Value> {
    let (mut ws, _) = tokio_tungstenite::connect_async(url.to_string())
        .await
        .unwrap();
    let mut decoded: Option<serde_json::Value> = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    while let Ok(Some(msg)) = tokio::time::timeout_at(deadline, ws.next()).await {
        let Ok(msg) = msg else { break };
        let Message::Text(text) = msg else { continue };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        match v.get("type").and_then(|t| t.as_str()) {
            Some("decoded") => decoded = Some(v),
            Some("done") => break,
            _ => {}
        }
    }
    let _ = ws.send(Message::Close(None)).await;
    decoded
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}
