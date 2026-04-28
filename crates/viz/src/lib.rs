//! Library face of the Javis visualisation server.
//!
//! Exposes the streaming pipeline and the wire types so integration
//! tests (and any future embedders) can run a session without going
//! through the binary.

pub mod events;
pub mod pipeline;
pub mod server;

pub use events::{DecodedWord, Event};
pub use pipeline::run_demo_session;
pub use server::{router, run_session};
