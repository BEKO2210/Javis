//! Encoders for Javis: turn symbolic input into Sparse Distributed
//! Representations (SDRs) that can be injected into the spiking network.
//!
//! Design follows Numenta's HTM / Semantic Folding spirit: every word
//! maps deterministically to a fixed-cardinality bag of bit indices in
//! a large, sparse address space (e.g. n=2048, k=20). Composition is
//! the union of token bags, so two phrases that share a word
//! automatically share that word's exact bits.
//!
//! Pure standard-library hashing (`DefaultHasher`), no external crates.
//!
//! # Quick start
//!
//! Encode two short phrases that share a word and verify the SDR
//! union contains the shared word's bits twice over:
//!
//! ```rust
//! use encoders::{Sdr, TextEncoder};
//!
//! let enc = TextEncoder::new(2048, 20);
//! let hello = enc.encode_word("hello");
//! let hello_world = enc.encode("hello world");
//! let hello_rust  = enc.encode("hello rust");
//!
//! // Both two-word SDRs contain the full hello-bag.
//! assert_eq!(hello_world.overlap(&hello), 20);
//! assert_eq!(hello_rust .overlap(&hello), 20);
//!
//! // The two together share at least the hello-bag (and very little else).
//! assert!(hello_world.overlap(&hello_rust) >= 20);
//! ```
//!
//! Stop-words can be filtered out so they do not contribute bits:
//!
//! ```rust
//! use encoders::TextEncoder;
//!
//! let stops = ["the", "is", "a"];
//! let enc = TextEncoder::with_stopwords(2048, 20, stops);
//! let plain = TextEncoder::new(2048, 20);
//!
//! // The encoder with a stop-list erases "the" and "is" entirely.
//! assert_eq!(
//!     enc  .encode("the cat is on the mat").indices,
//!     plain.encode("cat on mat").indices,
//! );
//! ```

pub mod decode;
pub mod inject;
pub mod sdr;
pub mod text;

pub use decode::EngramDictionary;
pub use inject::inject_sdr;
pub use sdr::Sdr;
pub use text::TextEncoder;
