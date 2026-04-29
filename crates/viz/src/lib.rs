//! Library face of the Javis visualisation server.
//!
//! Holds a single persistent brain in [`state::AppState`] and exposes
//! it via the WebSocket router from [`server`]. Integration tests use
//! the same building blocks the binary uses.

pub mod events;
pub mod metrics;
pub mod server;
pub mod state;

pub use events::{DecodedWord, Event};
pub use server::{router, router_no_static};
pub use state::AppState;
