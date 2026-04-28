//! Scaling benchmark on a small Wikipedia-shaped corpus.
//!
//! Five short paragraphs about five distinct topics (volcano, bicycle,
//! coffee, photosynthesis, Eiffel Tower) — enough vocabulary that the
//! brain has to keep multiple engrams separable while still answering
//! recall queries efficiently.
//!
//! What we measure
//! ---------------
//! - Per-query token reduction vs. naive RAG
//! - Per-query whether the queried keyword survives in the candidates
//! - Aggregates over all five queries: mean / min reduction
//!
//! What we don't measure here (deliberately)
//! -----------------------------------------
//! - Cross-domain associative recall (e.g. "energy" → both volcano
//!   and photosynthesis). That's a separate test below.

use eval::token_efficiency::{run_benchmark_on, BenchmarkResult};
use eval::{wiki_corpus, wiki_queries};

fn pretty(r: &BenchmarkResult) -> String {
    let words: Vec<String> = r
        .javis_words
        .iter()
        .map(|(w, s)| format!("{w}({s:.2})"))
        .collect();
    format!(
        "  query={:<14} rag={:>3}  javis={:>3}  saving={:>5.1}%  decoded=[{}]",
        format!("'{}'", r.query),
        r.rag_tokens,
        r.javis_tokens,
        r.token_reduction_pct,
        words.join(", "),
    )
}

#[test]
fn javis_scales_to_a_five_topic_wiki_corpus() {
    let corpus = wiki_corpus();
    let queries = wiki_queries();

    eprintln!("\nJavis vs naive RAG on a 5-topic Wikipedia-shaped corpus:");
    let mut results: Vec<BenchmarkResult> = Vec::new();
    for (topic, query) in queries {
        let r = run_benchmark_on(corpus, query);
        eprintln!("{}", pretty(&r));
        // Per-query contract: RAG retrieved something, the query word
        // is in the decoded set, and reduction is meaningful.
        assert!(r.rag_tokens > 0, "rag missed the topic '{topic}'");
        let words: Vec<&str> = r.javis_words.iter().map(|(w, _)| w.as_str()).collect();
        assert!(
            words.contains(query),
            "query '{query}' missing from decoded set: {words:?}",
        );
        assert!(
            r.token_reduction_pct >= 70.0,
            "reduction too low for '{query}': {:.1}%",
            r.token_reduction_pct,
        );
        results.push(r);
    }

    let n = results.len() as f32;
    let mean = results.iter().map(|r| r.token_reduction_pct).sum::<f32>() / n;
    let min = results
        .iter()
        .map(|r| r.token_reduction_pct)
        .fold(100.0_f32, f32::min);
    let max = results
        .iter()
        .map(|r| r.token_reduction_pct)
        .fold(0.0_f32, f32::max);
    eprintln!(
        "\naggregate: mean={:.1}%  min={:.1}%  max={:.1}%  ({} queries)",
        mean, min, max, results.len()
    );
    // Aggregate contract: even the worst query saves ≥ 70 % over RAG,
    // and the average saving is high.
    assert!(min >= 70.0, "aggregate min below threshold: {:.1}%", min);
    assert!(mean >= 80.0, "aggregate mean below threshold: {:.1}%", mean);
}

#[test]
fn engrams_remain_separable_after_five_topic_training() {
    // For each topic, run the recall and assert the *only* keyword
    // that survives the decode threshold is the one belonging to that
    // topic. With well-tuned plasticity this should hold across all
    // five sibling topics; if iSTDP / homeostasis fail, sibling words
    // will leak in.
    let corpus = wiki_corpus();
    let queries = wiki_queries();
    let topic_keywords: Vec<&str> = queries.iter().map(|(t, _)| *t).collect();

    let mut violations = Vec::<String>::new();
    for (topic, query) in queries {
        let r = run_benchmark_on(corpus, query);
        let decoded_words: Vec<&str> =
            r.javis_words.iter().map(|(w, _)| w.as_str()).collect();
        // Are any *other* topic keywords leaking in?
        for other in &topic_keywords {
            if other == topic {
                continue;
            }
            if decoded_words.contains(other) {
                violations.push(format!(
                    "query '{}' leaked '{}' (decoded: {:?})",
                    query, other, decoded_words,
                ));
            }
        }
    }

    if !violations.is_empty() {
        eprintln!("separability violations:");
        for v in &violations {
            eprintln!("  {v}");
        }
    }
    assert!(
        violations.is_empty(),
        "{} cross-topic leaks detected",
        violations.len(),
    );
}
