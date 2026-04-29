//! Tests for the WebSocket session concurrency cap.
//!
//! When the running session count reaches `JAVIS_MAX_CONCURRENT_SESSIONS`,
//! the upgrade handler should refuse the WebSocket upgrade with
//! `503 Service Unavailable` plus a `Retry-After` header. As soon as
//! one session ends, the freed permit lets the next client through.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upgrade_rejected_at_cap_zero() {
    // Cap = 0: every upgrade attempt is rejected. Sidesteps any
    // timing assumptions about session lifetime.
    let (addr, server) = spawn_server(0).await;

    let resp = ws_upgrade_request(addr).await;
    assert_eq!(
        resp.status,
        503,
        "expected 503 when cap is zero, got {} (body={:?})",
        resp.status,
        std::str::from_utf8(&resp.body).ok(),
    );
    assert_eq!(
        find_header(&resp, "retry-after"),
        Some("1"),
        "expected Retry-After: 1 header on rejection",
    );

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn released_permit_unblocks_next_session() {
    // Cap = 1, first session is a `train` (~700 ms wall): plenty of
    // time for a second upgrade to land in the busy window. After
    // the train completes, the permit is freed and a third upgrade
    // must succeed.
    viz::metrics::init();
    let (addr, server) = spawn_server(1).await;

    // First: kick off a long-ish train. Spawn so we can fire the
    // second request without waiting for the train to finish.
    let train_url = format!(
        "ws://{addr}/ws?action=train&text={}",
        urlencode("Volcanoes erupt when magma chambers pressurise and the rock above gives way."),
    );
    let train_handle = tokio::task::spawn(async move {
        use futures_util::StreamExt;
        let (mut ws, _) = tokio_tungstenite::connect_async(train_url).await.unwrap();
        // Drain the whole training event stream.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        while let Ok(Some(msg)) = tokio::time::timeout_at(deadline, ws.next()).await {
            if let Ok(tokio_tungstenite::tungstenite::Message::Text(txt)) = msg {
                if txt.contains("\"type\":\"done\"") {
                    break;
                }
            } else {
                break;
            }
        }
    });

    // Give the train session enough head start to lock the permit.
    tokio::time::sleep(Duration::from_millis(80)).await;

    // Second: must be rejected.
    let resp = ws_upgrade_request(addr).await;
    assert_eq!(
        resp.status, 503,
        "expected 503 while cap-1 train holds the permit, got {}",
        resp.status,
    );

    // Wait for the train to finish, releasing the permit.
    train_handle.await.unwrap();
    // Brief settle for the permit Drop to land.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Third: must be accepted now.
    let resp = ws_upgrade_request(addr).await;
    assert_eq!(
        resp.status, 101,
        "expected 101 (Switching Protocols) after permit release, got {}",
        resp.status,
    );

    // The metric should have at least one rejection recorded.
    let metrics = http_get_text(addr, "/metrics").await;
    assert!(
        metrics.lines().any(|l| {
            l.starts_with("javis_ws_rejected_total")
                && l.contains("reason=\"concurrency_cap\"")
                && l.split_whitespace()
                    .last()
                    .and_then(|n| n.parse::<u64>().ok())
                    .map(|n| n >= 1)
                    .unwrap_or(false)
        }),
        "expected javis_ws_rejected_total >= 1 with reason=concurrency_cap, body:\n{metrics}",
    );

    server.abort();
}

// ----- shared helpers ------------------------------------------------

async fn spawn_server(cap: usize) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(viz::AppState::with_session_cap(cap));
    let app = viz::server::router_no_static(state);
    let handle = tokio::task::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, handle)
}

struct RawResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn find_header<'a>(resp: &'a RawResponse, name: &str) -> Option<&'a str> {
    resp.headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Speak HTTP/1.1 directly so we can observe the status code on a
/// non-101 response — `tokio-tungstenite` would surface a 503 as a
/// handshake error and lose the headers we care about.
async fn ws_upgrade_request(addr: SocketAddr) -> RawResponse {
    let req = format!(
        "GET /ws?action=recall&query=rust HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Version: 13\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         \r\n",
    );
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();

    let mut buf = Vec::with_capacity(4096);
    let _ = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let mut tmp = [0u8; 4096];
            match stream.read(&mut tmp).await {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
    })
    .await;
    parse_http_response(&buf)
}

async fn http_get_text(addr: SocketAddr, path: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n",);
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[(split + 4).min(buf.len())..]).into_owned()
}

fn parse_http_response(buf: &[u8]) -> RawResponse {
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or(buf.len());
    let header_block = &buf[..split];
    let body_start = (split + 4).min(buf.len());
    let body = buf[body_start..].to_vec();

    let mut lines = std::str::from_utf8(header_block).unwrap().lines();
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    RawResponse {
        status,
        headers,
        body,
    }
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
