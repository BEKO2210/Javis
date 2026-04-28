//! SDR overlap properties.
//!
//! "hello world" and "hello rust" must share the encoding of "hello"
//! exactly, because each word maps deterministically to its own
//! `k` indices and the text SDR is the union of its tokens.
//! "foo bar" must overlap them only at chance level (≈ k²/n).

use encoders::{Sdr, TextEncoder};

const N: u32 = 2048;
const K: u32 = 20;

#[test]
fn shared_word_yields_exact_overlap() {
    let enc = TextEncoder::new(N, K);

    let hello = enc.encode_word("hello");
    let hw = enc.encode("hello world");
    let hr = enc.encode("hello rust");
    let fb = enc.encode("foo bar");

    // Both texts must contain every bit of "hello".
    assert_eq!(hw.overlap(&hello), K as usize, "hello word missing in 'hello world'");
    assert_eq!(hr.overlap(&hello), K as usize, "hello word missing in 'hello rust'");

    // The overlap of the two texts is at least the "hello" part.
    let hw_hr = hw.overlap(&hr);
    assert!(
        hw_hr >= K as usize,
        "expected overlap ≥ k={K}, got {hw_hr}",
    );

    // Above K we only allow a small chance overlap from random bit
    // collisions between "world" and "rust". Expectation k²/n ≈ 0.2.
    let extra = hw_hr - K as usize;
    assert!(extra <= 3, "spurious overlap above 'hello': {extra} bits");

    // "foo bar" shares no semantically meaningful overlap; any common
    // bits are pure chance and should be minor.
    let chance_hw_fb = hw.overlap(&fb);
    let chance_hr_fb = hr.overlap(&fb);
    assert!(
        chance_hw_fb <= 3,
        "unrelated texts overlap too much: hw vs fb = {chance_hw_fb}",
    );
    assert!(
        chance_hr_fb <= 3,
        "unrelated texts overlap too much: hr vs fb = {chance_hr_fb}",
    );

    // Sanity: every word always produces exactly k indices.
    assert_eq!(hello.len(), K as usize);
    assert_eq!(enc.encode_word("world").len(), K as usize);
    assert_eq!(enc.encode_word("rust").len(), K as usize);
}

#[test]
fn deterministic_across_calls() {
    let enc = TextEncoder::new(N, K);
    let a = enc.encode("the quick brown fox");
    let b = enc.encode("the quick brown fox");
    assert_eq!(a.indices, b.indices);
}

#[test]
fn punctuation_is_stripped() {
    let enc = TextEncoder::new(N, K);
    let a = enc.encode("hello, world!");
    let b = enc.encode("hello world");
    assert_eq!(a.indices, b.indices, "punctuation should not change the SDR");
}

#[test]
fn case_is_normalised() {
    let enc = TextEncoder::new(N, K);
    let a = enc.encode("Hello WORLD");
    let b = enc.encode("hello world");
    assert_eq!(a.indices, b.indices);
}

#[test]
fn stopwords_are_dropped() {
    let stops = ["the", "is", "on", "a"];
    let enc = TextEncoder::with_stopwords(N, K, stops);
    let plain = TextEncoder::new(N, K);

    let with_noise = enc.encode("the cat is on the mat");
    let cleaned = plain.encode("cat mat");
    assert_eq!(
        with_noise.indices, cleaned.indices,
        "stop words must contribute zero bits",
    );
    assert!(enc.is_stopword("THE"));
    assert!(!enc.is_stopword("cat"));
}

#[test]
fn union_and_overlap_agree_on_simple_case() {
    let n = 16;
    let a = Sdr::from_indices(n, vec![1, 3, 5, 7]);
    let b = Sdr::from_indices(n, vec![3, 5, 9, 11]);
    assert_eq!(a.overlap(&b), 2);
    let u = a.union(&b);
    assert_eq!(u.indices, vec![1, 3, 5, 7, 9, 11]);
}
