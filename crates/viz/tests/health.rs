//! HTTP probe tests for the `/health` (liveness) and `/ready`
//! (readiness) endpoints. Drives the router via `tower::oneshot`
//! instead of binding a real TCP socket — `axum::Router` is itself a
//! `tower::Service`, so we can hand it requests directly.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

#[tokio::test]
async fn health_returns_200_immediately() {
    let state = Arc::new(viz::AppState::new());
    let app = viz::server::router_no_static(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"ok");
}

#[tokio::test]
async fn ready_reports_brain_stats_as_json() {
    let state = Arc::new(viz::AppState::new_with_mock_llm());
    state
        .run_train("Lava is liquid molten rock from a volcano.".into(), None)
        .await;

    let app = viz::server::router_no_static(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
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
        content_type.starts_with("application/json"),
        "expected JSON content-type, got {content_type:?}",
    );

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v.get("status").and_then(|s| s.as_str()), Some("ready"));
    assert_eq!(v.get("sentences").and_then(|n| n.as_u64()), Some(1));
    assert!(v.get("words").and_then(|n| n.as_u64()).unwrap_or(0) > 0);
    assert_eq!(v.get("llm").and_then(|s| s.as_str()), Some("mock"));
}

#[tokio::test]
async fn ready_works_on_empty_brain() {
    // An empty brain right after construction is still "ready" — the
    // process can serve recall/train/ask requests, the recall just won't
    // surface anything yet. Probes must not flap during cold start.
    let state = Arc::new(viz::AppState::new_with_mock_llm());
    let app = viz::server::router_no_static(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v.get("sentences").and_then(|n| n.as_u64()), Some(0));
    assert_eq!(v.get("words").and_then(|n| n.as_u64()), Some(0));
}
