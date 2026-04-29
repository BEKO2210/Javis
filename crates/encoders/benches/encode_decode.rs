//! Benchmarks for the text → SDR encoder and the SDR → words decoder.
//! These two are the per-request entry/exit points of every recall —
//! together they bound how fast a single decode round can run no
//! matter how cheap the spike simulation gets.
//!
//! Run locally:
//!   `cargo bench -p encoders --bench encode_decode`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use encoders::{EngramDictionary, TextEncoder};

const ENC_N: u32 = 1000;
const ENC_K: u32 = 20;

const STOPWORDS: &[&str] = &[
    "is", "a", "the", "an", "on", "at", "of", "in", "to", "and", "or", "for", "with", "by", "from",
    "but", "as", "it", "its", "this", "that", "these", "those", "be", "are", "was", "were", "like",
];

fn fresh_encoder() -> TextEncoder {
    TextEncoder::with_stopwords(ENC_N, ENC_K, STOPWORDS.iter().copied())
}

const SHORT: &str = "rust";
const LONG: &str = "Rust is a systems programming language focused on memory safety \
     and ownership; the borrow checker prevents data races at compile time.";

fn bench_encode_word(c: &mut Criterion) {
    let enc = fresh_encoder();
    c.bench_function("encode_word", |b| {
        b.iter(|| {
            let s = enc.encode_word(black_box(SHORT));
            black_box(s);
        });
    });
}

fn bench_encode_sentence(c: &mut Criterion) {
    let enc = fresh_encoder();
    c.bench_function("encode_sentence", |b| {
        b.iter(|| {
            let s = enc.encode(black_box(LONG));
            black_box(s);
        });
    });
}

/// Build a dictionary populated with `n_words` synthetic concepts so
/// `decode` has a realistic vocabulary to scan against.
fn populated_dict(n_words: usize) -> (EngramDictionary, Vec<u32>) {
    let mut dict = EngramDictionary::new();
    // Each concept gets a kWTA-style index set of size 200 — same
    // size the live AppState uses (`KWTA_K`).
    let kwta_k = 200u32;
    for i in 0..n_words {
        let base = (i as u32) % 1000;
        let indices: Vec<u32> = (0..kwta_k).map(|j| (base + j) % 2000).collect();
        dict.learn_concept(&format!("word_{i}"), &indices);
    }
    // A query pattern that overlaps with the first concept by ~50 %.
    let query: Vec<u32> = (0..kwta_k).map(|j| j % 2000).collect();
    (dict, query)
}

fn bench_decode_strict(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_strict");
    for &n in &[10usize, 100, 1000] {
        let (dict, query) = populated_dict(n);
        group.bench_function(format!("vocab_{n}"), |b| {
            b.iter(|| {
                let v = dict.decode(black_box(&query), 0.3);
                black_box(v);
            });
        });
    }
    group.finish();
}

fn bench_decode_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_top_k");
    for &n in &[10usize, 100, 1000] {
        let (dict, query) = populated_dict(n);
        group.bench_function(format!("vocab_{n}"), |b| {
            b.iter(|| {
                let v = dict.decode_top(black_box(&query), 5);
                black_box(v);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_word,
    bench_encode_sentence,
    bench_decode_strict,
    bench_decode_top_k,
);
criterion_main!(benches);
