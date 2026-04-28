//! Wire format for the WebSocket stream.
//!
//! One enum, one JSON message per `Event`. The browser deserialises
//! these and animates the brain. Schema is kept flat and small so we
//! can ship many per second without choking the socket.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Static topology: how big each region is, sent once at session start.
    Init {
        r1_size: u32,
        r2_size: u32,
        r2_excitatory: u32,
        r2_inhibitory: u32,
    },
    /// A new phase of the demo started (training / cooldown / recall / decode).
    Phase {
        name: String,
        detail: String,
    },
    /// One simulated millisecond's worth of spikes, batched.
    /// `r1` and `r2` carry neuron indices that fired during the batch.
    Step {
        t_ms: f32,
        r1: Vec<u32>,
        r2: Vec<u32>,
    },
    /// Result of decoding a recall pattern back to text candidates.
    Decoded {
        query: String,
        candidates: Vec<DecodedWord>,
        rag_tokens: u32,
        javis_tokens: u32,
        reduction_pct: f32,
        rag_payload: String,
        javis_payload: String,
    },
    /// Both LLM calls (RAG-payload-context and Javis-payload-context)
    /// have come back with answers and token usage.
    Asked {
        question: String,
        rag: LlmReply,
        javis: LlmReply,
    },
    /// Demo session has finished.
    Done,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmReply {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// True if a real Anthropic API call returned this; false for the
    /// deterministic offline mock.
    pub real: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodedWord {
    pub word: String,
    pub score: f32,
}
