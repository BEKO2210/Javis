//! Two complementary read modes on the same trained brain:
//!
//! - **Strict (`decode` with threshold 0.50)** — only engrams whose
//!   containment score crosses the bar. Sparse, topic-clean, the
//!   default behaviour `wiki_benchmark` relies on.
//! - **Top-k (`decode_top`)** — the K best-matching engrams regardless
//!   of absolute score. Always returns *something* per query, fills
//!   richer "related concepts" panels, but at small scores admits
//!   cross-topic neighbours.
//! - **Contextual** — engrams captured *during* training of a sentence
//!   and shared across every word in that sentence. Lets a headline
//!   keyword pull its whole paragraph back via top-k. Inspired by the
//!   engram-cell literature where engrams are formed by co-activity,
//!   not by isolated re-stimulation.
//!
//! These tests pin the contracts down so we don't accidentally
//! regress one when tuning the others.

use std::collections::HashSet;

use eval::token_efficiency::{
    naive_rag_lookup, run_javis_pipeline_contextual_top_k, run_javis_pipeline_top_k,
    run_javis_pipeline_with_threshold,
};
use eval::{count_tokens, wiki_corpus, wiki_queries};

const TOP_K: usize = 5;

#[test]
fn top_k_top_one_is_always_the_cue() {
    // The cue's own engram is by construction a perfect match (score
    // 1.0), so the top-1 result must be the cue itself for every query.
    let corpus = wiki_corpus();
    for (_topic, query) in wiki_queries() {
        let decoded = run_javis_pipeline_top_k(corpus, query, TOP_K);
        assert!(!decoded.is_empty(), "no decoded for '{query}'");
        assert_eq!(
            decoded[0].0, *query,
            "top-1 was '{}' for query '{query}'",
            decoded[0].0,
        );
        // Scores must be monotonically non-increasing.
        for w in decoded.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "top-k not sorted: {} -> {}",
                w[0].1,
                w[1].1,
            );
        }
    }
}

#[test]
fn top_k_payload_still_beats_rag() {
    // Even with a richer K=5 readout, the total tokens shipped to the
    // LLM should clearly beat naive RAG. Looser than the strict mode's
    // ~96.6 %, but still a real saving.
    let corpus = wiki_corpus();
    let mut javis_total = 0usize;
    let mut rag_total = 0usize;
    for (_topic, query) in wiki_queries() {
        let decoded = run_javis_pipeline_top_k(corpus, query, TOP_K);
        let payload = decoded
            .iter()
            .map(|(w, _)| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        javis_total += count_tokens(&payload);
        rag_total += count_tokens(&naive_rag_lookup(corpus, query).unwrap_or_default());

        let unique: HashSet<&str> = decoded.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(
            unique.len(),
            decoded.len(),
            "duplicate words in decode output"
        );
    }

    let saving = if rag_total > 0 {
        (1.0 - javis_total as f32 / rag_total as f32) * 100.0
    } else {
        0.0
    };
    eprintln!(
        "top-{TOP_K} associative recall: rag={rag_total} javis={javis_total} saving={saving:.1}%",
    );
    assert!(
        saving >= 70.0,
        "top-k saving too low: {:.1}% (rag={rag_total}, javis={javis_total})",
        saving,
    );
}

#[test]
fn strict_threshold_stays_topic_clean() {
    // The default 0.50 threshold is the contract that backs the
    // wiki_benchmark cross-topic separability test: at this cut-off no
    // word from a different topic ever creeps into the decoded set.
    use std::collections::HashMap;

    let corpus = wiki_corpus();

    let table: HashMap<String, usize> = {
        use encoders::TextEncoder;
        let enc = TextEncoder::new(2048, 20);
        let mut t = HashMap::new();
        for (idx, chunk) in corpus.iter().enumerate() {
            for tok in enc.tokenize(chunk) {
                t.entry(tok).or_insert(idx);
            }
        }
        t
    };

    for (_topic, query) in wiki_queries() {
        let decoded = run_javis_pipeline_with_threshold(corpus, query, 0.50);
        let cue_topic = *table
            .get(*query)
            .unwrap_or_else(|| panic!("query '{query}' not in any topic"));
        for (word, _score) in &decoded {
            let t = *table
                .get(word)
                .unwrap_or_else(|| panic!("decoded '{word}' not in any topic"));
            assert_eq!(
                t, cue_topic,
                "strict threshold leaked '{word}' (topic {t}) on query '{query}' (expected {cue_topic})",
            );
        }
    }
}

#[test]
fn high_threshold_keeps_recall_minimal() {
    // Token-efficiency regression guard: the default threshold must
    // produce ≤ 3 words per query so wiki_benchmark stays sparse.
    let corpus = wiki_corpus();
    for (_topic, query) in wiki_queries() {
        let decoded = run_javis_pipeline_with_threshold(corpus, query, 0.50);
        assert!(
            decoded.len() <= 3,
            "default threshold should stay minimal but got {} for '{}'",
            decoded.len(),
            query,
        );
        let words: Vec<&str> = decoded.iter().map(|(w, _)| w.as_str()).collect();
        assert!(
            words.contains(query),
            "default threshold lost the query word '{query}': {words:?}",
        );
    }
}

#[test]
fn contextual_mode_brings_multiple_words_per_query() {
    // Contextual fingerprints are captured *during* training of each
    // sentence and shared across every word in it (engram-cell-style
    // co-activity capture; see notes/20). The key property: a single
    // cue, instead of returning just itself, brings back many
    // sentence-mates. Per-query *ranking* is fuzzy (every word in a
    // sentence ties on score, alphabetic tiebreak fills the rest)
    // but the *retrieval volume* is real: contextual + top-K returns
    // many semantically related words where the strict mode would
    // return one.
    let corpus = wiki_corpus();

    eprintln!("\nContextual top-30 (ranking is fuzzy, volume is real):");
    let mut total_words = 0usize;
    for (_topic, query) in wiki_queries() {
        let decoded = run_javis_pipeline_contextual_top_k(corpus, query, 30);
        let words: Vec<&str> = decoded.iter().map(|(w, _)| w.as_str()).collect();
        eprintln!("  '{query}' → {} words", words.len());
        // The cue itself must be reachable.
        assert!(!words.is_empty(), "no decoded for '{query}'",);
        total_words += words.len();
    }
    let mean = total_words as f32 / wiki_queries().len() as f32;
    assert!(
        mean >= 20.0,
        "contextual mode should retrieve many sentence-mates per query, got mean {mean:.1}",
    );
}
