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
async fn ask_returns_both_answers_in_mock_mode() {
    // Force the LLM into mock mode so the test never hits the network.
    std::env::remove_var("ANTHROPIC_API_KEY");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(viz::AppState::new_with_mock_llm());
    let app = viz::server::router_no_static(state);
    let server = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let url = format!(
        "ws://{addr}/ws?action=ask&query=rust&rag={}&javis=rust",
        urlencode("Rust is a systems language focused on memory safety."),
    );

    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    let mut asked: Option<serde_json::Value> = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while let Ok(Some(msg)) = tokio::time::timeout_at(deadline, ws.next()).await {
        let Ok(msg) = msg else { break };
        let Message::Text(text) = msg else { continue };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        match v.get("type").and_then(|t| t.as_str()) {
            Some("asked") => asked = Some(v),
            Some("done") => break,
            _ => {}
        }
    }
    let _ = ws.send(Message::Close(None)).await;

    let a = asked.expect("no asked event received");
    let rag = a.get("rag").unwrap();
    let javis = a.get("javis").unwrap();
    assert!(rag.get("text").and_then(|t| t.as_str()).unwrap().len() > 0);
    assert!(javis.get("text").and_then(|t| t.as_str()).unwrap().len() > 0);
    assert_eq!(rag.get("real").and_then(|v| v.as_bool()), Some(false));
    assert_eq!(javis.get("real").and_then(|v| v.as_bool()), Some(false));
    let rag_in = rag.get("input_tokens").and_then(|v| v.as_u64()).unwrap();
    let jvs_in = javis.get("input_tokens").and_then(|v| v.as_u64()).unwrap();
    assert!(
        rag_in > jvs_in,
        "expected RAG context to use more tokens than Javis: rag={rag_in} javis={jvs_in}",
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_round_trip_preserves_recall() {
    // 1) Train on a sentence, recall it, capture the candidate set.
    let state_a = Arc::new(viz::AppState::new_with_mock_llm());
    state_a
        .run_train(
            "Lava is liquid molten rock from a volcano.".into(),
            None,
        )
        .await;
    let (s_before, w_before) = state_a.stats().await;
    assert!(s_before == 1 && w_before > 0);

    // 2) Save to a temp file.
    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join(format!(
        "javis-snapshot-{}.json",
        std::process::id()
    ));
    state_a.save_to_file(&path).await.unwrap();
    assert!(path.exists());

    // 3) Build a fresh, empty AppState and load the snapshot.
    let state_b = Arc::new(viz::AppState::new_with_mock_llm());
    let (s_empty, w_empty) = state_b.stats().await;
    assert_eq!((s_empty, w_empty), (0, 0));
    state_b.load_from_file(&path).await.unwrap();
    let (s_after, w_after) = state_b.stats().await;
    assert_eq!((s_after, w_after), (s_before, w_before));

    // 4) Recall a word from the trained sentence on the *loaded* brain
    //    and confirm it surfaces in the candidate list.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = viz::server::router_no_static(state_b);
    let server = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let url = format!("ws://{addr}/ws?action=recall&query=lava");
    let decoded = walk_until_done(&url).await;
    let d = decoded.expect("no decoded after snapshot load");
    let candidates = d.get("candidates").and_then(|v| v.as_array()).unwrap();
    let words: Vec<String> = candidates
        .iter()
        .filter_map(|c| c.get("word").and_then(|w| w.as_str()).map(str::to_string))
        .collect();
    assert!(
        words.iter().any(|w| w == "lava"),
        "loaded brain lost the trained word: {words:?}",
    );

    server.abort();
    let _ = std::fs::remove_file(&path);
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
