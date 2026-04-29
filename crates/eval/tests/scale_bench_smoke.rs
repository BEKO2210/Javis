//! Smoke test for the scale-benchmark plumbing.
//!
//! Runs the corpus generator + the train-once / query-many harness
//! at a tiny size so the test suite can verify the end-to-end wiring
//! without paying the multi-minute training cost of the real
//! benchmark. Asserts only on contract-level invariants — exact
//! precision / recall numbers vary with seed and brain-size; for
//! quality validation use `cargo run -p eval --example scale_benchmark`.

use eval::{build_scale_corpus, ScaleBrain};

#[test]
fn scale_benchmark_round_trip() {
    // Tiny: 16 sentences ≈ 8 s of training. Keeps the test under the
    // CI budget while still exercising every code path.
    let corpus = build_scale_corpus(16, 7);
    assert!(!corpus.sentences.is_empty());
    assert!(!corpus.vocabulary.is_empty());

    let mut brain = ScaleBrain::train_on(&corpus);
    assert_eq!(brain.n_sentences, 16);
    assert_eq!(brain.vocab_size, corpus.vocabulary.len());
    assert!(brain.training_secs > 0.0);

    // Pick a handful of queries deterministically and run them.
    let queries: Vec<String> = corpus.queries.iter().take(5).cloned().collect();
    assert!(
        !queries.is_empty(),
        "tiny corpus should still yield ≥ 1 query candidate",
    );

    let report = brain.evaluate(&queries, 6);
    assert_eq!(report.summary.n_queries, queries.len());
    assert_eq!(report.per_query.len(), queries.len());

    // Precision contract: every query word that the encoder maps to
    // a non-empty SDR *should* end up in its own decoded set —
    // otherwise the engram dictionary lost the concept it just
    // learned. Empty-SDR queries (very short words) are exempt.
    for r in &report.per_query {
        if r.decoded.is_empty() {
            continue;
        }
        assert!(
            r.has_self,
            "query '{}' did not surface itself in the decoded set: {:?}",
            r.query, r.decoded,
        );
    }

    // Markdown rendering should never panic and produce a non-empty
    // header.
    let md = report.render_markdown();
    assert!(md.contains("Scale benchmark summary"));
    assert!(md.contains("Mean token reduction"));
}
