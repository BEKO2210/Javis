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

pub mod sdr;
pub mod text;
pub mod inject;
pub mod decode;

pub use sdr::Sdr;
pub use text::TextEncoder;
pub use inject::inject_sdr;
pub use decode::EngramDictionary;
