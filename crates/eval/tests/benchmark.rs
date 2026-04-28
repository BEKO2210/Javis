//! End-to-end token efficiency benchmark.
//!
//! Runs a query through both pipelines, prints both payloads and the
//! token saving, and asserts that Javis returns substantially less
//! text than naive RAG while still surfacing the right concepts.

use eval::token_efficiency::{run_benchmark, BenchmarkResult};

fn assert_reduces_tokens(r: &BenchmarkResult) {
    assert!(
        r.rag_tokens > 0,
        "RAG returned an empty payload — query missed the corpus",
    );
    assert!(
        !r.javis_words.is_empty(),
        "Javis returned no candidates for query '{}'",
        r.query,
    );
    // The payoff line: at least 70 % fewer tokens than naive RAG.
    assert!(
        r.token_reduction_pct >= 70.0,
        "expected ≥ 70 % token reduction, got {:.1}% (rag={}, javis={})",
        r.token_reduction_pct,
        r.rag_tokens,
        r.javis_tokens,
    );
}

#[test]
fn javis_reduces_tokens_for_rust_query() {
    let r = run_benchmark("rust");
    eprintln!("{}", r.report());

    // The recall must include the query keyword itself.
    let words: Vec<&str> = r.javis_words.iter().map(|(w, _)| w.as_str()).collect();
    assert!(
        words.contains(&"rust"),
        "Javis recall lost the query word: {:?}",
        words,
    );
    assert_reduces_tokens(&r);
}

#[test]
fn javis_reduces_tokens_for_python_query() {
    let r = run_benchmark("python");
    eprintln!("{}", r.report());
    let words: Vec<&str> = r.javis_words.iter().map(|(w, _)| w.as_str()).collect();
    assert!(words.contains(&"python"), "lost query word: {:?}", words);
    assert_reduces_tokens(&r);
}

#[test]
fn javis_reduces_tokens_for_cpp_query() {
    let r = run_benchmark("cpp");
    eprintln!("{}", r.report());
    let words: Vec<&str> = r.javis_words.iter().map(|(w, _)| w.as_str()).collect();
    assert!(words.contains(&"cpp"), "lost query word: {:?}", words);
    assert_reduces_tokens(&r);
}
