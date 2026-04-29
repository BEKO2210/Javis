//! Smallest possible Javis demo.
//!
//! Run with:
//!     cargo run --release -p eval --example hello_javis
//!
//! Trains the brain on three Wikipedia-shaped paragraphs, queries with
//! every topic keyword, prints the RAG-vs-Javis comparison + saving.
//! No network, no API keys, no UI — just text in, decoded concepts +
//! token numbers out.

use eval::token_efficiency::{naive_rag_lookup, run_javis_pipeline};
use eval::{count_tokens, wiki_corpus, wiki_queries};

fn main() {
    let corpus = wiki_corpus();
    let queries = wiki_queries();

    let mut total_rag = 0usize;
    let mut total_javis = 0usize;

    println!("Javis demo — five Wikipedia-shaped paragraphs\n");
    println!(
        "{:<14}  {:>8}  {:>8}  {:>8}   decoded",
        "query", "rag", "javis", "saving",
    );
    println!("{}", "─".repeat(70));

    for (_topic, query) in queries {
        let rag = naive_rag_lookup(corpus, query).unwrap_or_default();
        let rag_tok = count_tokens(&rag);

        let decoded = run_javis_pipeline(corpus, query);
        let javis_payload = decoded
            .iter()
            .map(|(w, _)| w.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let javis_tok = count_tokens(&javis_payload);

        let saving = if rag_tok > 0 {
            (1.0 - javis_tok as f32 / rag_tok as f32) * 100.0
        } else {
            0.0
        };

        println!(
            "{:<14}  {:>8}  {:>8}  {:>7.1}%   [{}]",
            format!("'{query}'"),
            rag_tok,
            javis_tok,
            saving,
            decoded
                .iter()
                .map(|(w, s)| format!("{w}:{s:.2}"))
                .collect::<Vec<_>>()
                .join(", "),
        );

        total_rag += rag_tok;
        total_javis += javis_tok;
    }

    println!("{}", "─".repeat(70));
    let total_saving = if total_rag > 0 {
        (1.0 - total_javis as f32 / total_rag as f32) * 100.0
    } else {
        0.0
    };
    println!(
        "{:<14}  {:>8}  {:>8}  {:>7.1}%",
        "total", total_rag, total_javis, total_saving,
    );
}
