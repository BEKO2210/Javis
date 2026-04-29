//! Prometheus metrics installation and exposition.
//!
//! The metrics recorder is a process-global singleton (the `metrics`
//! crate uses a global recorder pointer), so we install it exactly
//! once via [`init`] and stash the [`PrometheusHandle`] in a
//! [`OnceLock`] for the `/metrics` HTTP handler to render later.
//!
//! ## Metrics emitted
//!
//! | Name | Type | Labels | Meaning |
//! | --- | --- | --- | --- |
//! | `javis_ws_sessions_total` | counter | `action` | WebSocket sessions started |
//! | `javis_train_duration_seconds` | histogram | — | Wall time of a `train` call |
//! | `javis_recall_duration_seconds` | histogram | — | Wall time of a `recall` call |
//! | `javis_ask_duration_seconds` | histogram | `real` | Wall time of an `ask` call (real vs mock LLM) |
//! | `javis_snapshot_duration_seconds` | histogram | `op` | Wall time of save / load |
//! | `javis_brain_sentences` | gauge | — | Trained sentences currently in the brain |
//! | `javis_brain_words` | gauge | — | Distinct words in the dictionary |
//! | `javis_recall_tokens_rag_total` | counter | — | Σ RAG-context tokens billed across recalls |
//! | `javis_recall_tokens_javis_total` | counter | — | Σ Javis-context tokens billed across recalls |
//!
//! Modules emitting these (currently `viz::server` and `viz::state`)
//! call the matching `metrics::*!` macro directly — no wrappers, so
//! the recorder facade can keep them lock-free.

use std::sync::OnceLock;

use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Histogram bucket boundaries in seconds, sized for the operations we
/// run: a fast recall is 30-100 ms; a train is 700-900 ms; an LLM ask
/// can take seconds when real. The same bucket vector is applied to
/// every `*_duration_seconds` metric.
const DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Install the Prometheus recorder as the global metrics backend.
///
/// Idempotent: only the first caller actually installs; later calls
/// silently reuse the existing handle. Returns `true` if this call
/// performed the install. Tests that need a clean recorder slot can
/// rely on subsequent calls being no-ops.
pub fn init() -> bool {
    let mut installed = false;
    HANDLE.get_or_init(|| {
        installed = true;
        PrometheusBuilder::new()
            .set_buckets_for_metric(Matcher::Suffix("duration_seconds".into()), DURATION_BUCKETS)
            .expect("histogram bucket config")
            .install_recorder()
            .expect("install Prometheus recorder")
    });
    installed
}

/// Render the current metrics in the Prometheus exposition format.
///
/// Returns an empty string when [`init`] has not been called — useful
/// for tests that don't care about metrics, and for the `/metrics`
/// handler before the binary's first `init` call (which in practice
/// happens before the listener is up, so this case is unreachable).
pub fn render() -> String {
    HANDLE.get().map(|h| h.render()).unwrap_or_default()
}

/// Have we installed a recorder in this process?
pub fn is_initialised() -> bool {
    HANDLE.get().is_some()
}
