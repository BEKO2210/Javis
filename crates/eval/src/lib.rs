//! Javis evaluation harness.
//!
//! End-to-end benchmarks that compare Javis against alternative
//! retrieval pipelines. The first one — and the whole point of the
//! project — is the **token-efficiency** comparison: how many tokens
//! would each system actually send to a downstream LLM for a given
//! query?

pub mod scale_bench;
pub mod scale_corpus;
pub mod token_efficiency;
pub mod wiki_corpus;

pub use scale_bench::{Iter44Config, ScaleBrain, ScaleQueryResult, ScaleReport, ScaleSummary};
pub use scale_corpus::{build_scale_corpus, ScaleCorpus};
pub use wiki_corpus::{wiki_corpus, wiki_queries};

/// Crude token estimate. Modern BPE tokenisers (GPT, Claude) average
/// roughly 1.3 tokens per English whitespace-separated word, so we use
/// `ceil(words * 1.3)` as a deterministic heuristic. The same formula
/// is applied to every payload, so the *ratio* between RAG and Javis
/// is invariant to the constant.
pub fn count_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    if words == 0 {
        return 0;
    }
    ((words as f32) * 1.3).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_zero_tokens() {
        assert_eq!(count_tokens(""), 0);
        assert_eq!(count_tokens("   "), 0);
    }

    #[test]
    fn token_count_scales_with_words() {
        let one_word = count_tokens("hello");
        let ten_words = count_tokens(&"hello ".repeat(10));
        assert!(ten_words > one_word);
        assert_eq!(one_word, 2); // ceil(1 * 1.3) = 2
        assert_eq!(ten_words, 13); // ceil(10 * 1.3) = 13
    }
}
