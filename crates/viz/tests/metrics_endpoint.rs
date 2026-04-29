//! Prometheus metrics tests.
//!
//! The `metrics` crate uses a process-global recorder, so all tests in
//! this binary share one recorder installed by `viz::metrics::init()`.
//! That's why every test runs through this single binary's setup —
//! installing the recorder twice in different test binaries is fine
//! (each gets its own process), but installing twice in the same
//! process would fail.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn install_recorder() {
    // Idempotent — only the first call actually installs.
    viz::metrics::init();
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    install_recorder();

    // Drive at least one operation so there's something to expose.
    let state = Arc::new(viz::AppState::new_with_mock_llm());
    state
        .run_train("Lava is liquid molten rock from a volcano.".into(), None)
        .await;

    let app = viz::server::router_no_static(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/plain"),
        "expected text/plain, got {content_type:?}",
    );

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8(body.to_vec()).unwrap();

    // The recorder emits `# TYPE` headers for every registered metric.
    // We only assert on the names so tweaks to wording don't break the
    // test, but we do assert on every metric the spec promises.
    for name in [
        "javis_train_duration_seconds",
        "javis_brain_sentences",
        "javis_brain_words",
    ] {
        assert!(
            body.contains(name),
            "expected metric {name} in /metrics output, got:\n{body}",
        );
    }
}

#[tokio::test]
async fn metrics_endpoint_works_before_any_operation() {
    install_recorder();
    // Fresh state, no operations — the endpoint must still return 200
    // (Prometheus interprets empty body as "no metrics yet", not an
    // error).
    let state = Arc::new(viz::AppState::new_with_mock_llm());
    let app = viz::server::router_no_static(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn ws_session_counter_increments_on_each_session() {
    install_recorder();

    // Direct end-to-end: spin up a real router, hit it with a recall WS
    // request twice, then scrape metrics. The counter must show ≥ 2.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(viz::AppState::new_with_mock_llm());
    let app = viz::server::router_no_static(state.clone());
    let server = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    for _ in 0..2 {
        let url = format!("ws://{addr}/ws?action=recall&query=anything");
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        // Drain until close.
        use futures_util::StreamExt;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        while let Ok(Some(_msg)) = tokio::time::timeout_at(deadline, ws.next()).await {
            // ignore
        }
    }

    // Metrics scrape via the same router that serves /metrics.
    let app2 = viz::server::router_no_static(state);
    let resp = app2
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8(body.to_vec()).unwrap();

    // The counter line looks like:
    //   javis_ws_sessions_total{action="recall"} 2
    // We just check the family is present and a non-zero number sits
    // on at least one line for action="recall".
    assert!(
        body.lines()
            .any(|line| line.starts_with("javis_ws_sessions_total")
                && line.contains("action=\"recall\"")
                && line
                    .split_whitespace()
                    .last()
                    .and_then(|n| n.parse::<u64>().ok())
                    .map(|n| n >= 2)
                    .unwrap_or(false)),
        "expected javis_ws_sessions_total{{action=\"recall\"}} >= 2, body:\n{body}",
    );

    server.abort();
}
