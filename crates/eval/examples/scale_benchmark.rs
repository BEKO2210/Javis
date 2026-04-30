//! Validation-at-scale benchmark for Javis.
//!
//! Builds a deterministic template-driven corpus, trains the SNN
//! once, then evaluates a representative subset of vocabulary words
//! as queries. Reports mean token reduction, decoder precision /
//! recall / false-positive counts, and decoder latency — all in a
//! Markdown table that can be pasted into a release-notes page.
//!
//! Run:
//!
//! ```sh
//! # Quick smoke (30 sentences, ~15 s):
//! cargo run --release -p eval --example scale_benchmark -- --sentences 30
//!
//! # Default (200 sentences, ~3 minutes):
//! cargo run --release -p eval --example scale_benchmark
//!
//! # "Are we ready to claim this works at scale?" run:
//! cargo run --release -p eval --example scale_benchmark -- --sentences 500
//! ```
//!
//! By default the benchmark caps at 60 queries to keep wall-time
//! under five minutes; pass `--queries N` to override.

use std::time::Instant;

use eval::{build_scale_corpus, Iter44Config, ScaleBrain};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sentences = parse_arg(&args, "--sentences", 200);
    let queries_cap = parse_arg(&args, "--queries", 60);
    let decode_k = parse_arg(&args, "--decode-k", 6);
    let seed: u64 = parse_arg(&args, "--seed", 42);

    // iter-44 stack toggles.
    //   --iter44 off       (default) = pre-iter-44 baseline.
    //   --iter44 stability = heterosynaptic + metaplasticity only.
    //   --iter44 tuned     = intrinsic + heterosynaptic + structural,
    //                        params calibrated for short-corpus runs.
    //   --iter44 full      = every mechanism on (dev / stress test).
    let iter44 = match parse_string(&args, "--iter44").as_deref() {
        Some("full") => Iter44Config::full(),
        Some("stability") => Iter44Config::stability_only(),
        Some("tuned") => Iter44Config::tuned_for_short_corpus(),
        Some("off") | None => Iter44Config::iter43(),
        Some(other) => {
            eprintln!("--iter44 must be one of: off | stability | tuned | full (got '{other}')");
            std::process::exit(2);
        }
    };

    eprintln!(
        "Scale benchmark: sentences={sentences} seed={seed} queries_cap={queries_cap} \
         decode_k={decode_k} iter44={iter44:?}",
    );

    let t0 = Instant::now();
    eprintln!("[1/3] generating corpus …");
    let corpus = build_scale_corpus(sentences, seed);
    eprintln!(
        "  corpus: {} sentences, {} unique vocabulary words, {} candidate queries",
        corpus.sentences.len(),
        corpus.vocabulary.len(),
        corpus.queries.len(),
    );

    eprintln!("[2/3] training SNN (this is the slow step) …");
    let mut brain = ScaleBrain::train_on_with_config(&corpus, &iter44);
    eprintln!(
        "  trained in {:.1} s — vocab {} engrams in dictionary",
        brain.training_secs, brain.vocab_size,
    );

    // Subsample queries deterministically so a 200-sentence corpus
    // does not run 1500 queries by default.
    let queries: Vec<String> = corpus
        .queries
        .iter()
        .step_by((corpus.queries.len() / queries_cap).max(1))
        .take(queries_cap)
        .cloned()
        .collect();

    eprintln!("[3/3] evaluating {} queries …", queries.len());
    let report = brain.evaluate(&queries, decode_k);

    let total_secs = t0.elapsed().as_secs_f64();
    eprintln!("Total wall-time: {:.1} s", total_secs);
    eprintln!();

    // Markdown report goes to stdout so it can be redirected to a
    // file or piped into a CI annotation.
    print!("{}", report.render_markdown());
}

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == name {
            if let Ok(v) = args[i + 1].parse::<T>() {
                return v;
            }
        }
        i += 1;
    }
    default
}

fn parse_string(args: &[String], name: &str) -> Option<String> {
    let mut i = 0;
    while i + 1 < args.len() {
        if args[i] == name {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}
