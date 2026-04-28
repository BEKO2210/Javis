//! Claude API adapter for Javis.
//!
//! Two modes:
//!
//! - **Real**: when `ANTHROPIC_API_KEY` is set in the environment, the
//!   client posts to `https://api.anthropic.com/v1/messages` and returns
//!   the model's actual answer plus the actual token usage reported by
//!   the API.
//! - **Mock**: when no key is configured (CI, sandbox, offline demo)
//!   the client deterministically constructs an answer from the
//!   provided context. Token counts use the same `ceil(words * 1.3)`
//!   heuristic as the rest of Javis. Tests run in this mode.
//!
//! The same `ask(question, context)` API works in both modes, so the
//! UI and the server don't have to care.

use std::time::Duration;

use serde::{Deserialize, Serialize};

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-haiku-4-5-20251001";
const MAX_TOKENS: u32 = 256;

#[derive(Debug, Clone, Serialize)]
pub struct LlmAnswer {
    /// The model's reply (or a deterministic mock-text when offline).
    pub text: String,
    /// Actual prompt tokens reported by the API; in mock mode the
    /// `ceil(words * 1.3)` heuristic over the rendered prompt.
    pub input_tokens: u32,
    /// Actual response tokens reported by the API; mock heuristic
    /// otherwise.
    pub output_tokens: u32,
    /// Whether the answer came from a real API call.
    pub real: bool,
}

#[derive(Debug, Clone)]
pub struct LlmClient {
    api_key: Option<String>,
    model: String,
    timeout: Duration,
}

impl LlmClient {
    /// Build a client. Reads `ANTHROPIC_API_KEY` from the environment
    /// — present means real calls, absent means mock.
    pub fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok().filter(|s| !s.is_empty());
        Self {
            api_key,
            model: std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            timeout: Duration::from_secs(20),
        }
    }

    /// Force-mock client for tests / offline demos.
    pub fn mock() -> Self {
        Self {
            api_key: None,
            model: DEFAULT_MODEL.to_string(),
            timeout: Duration::from_secs(20),
        }
    }

    pub fn is_real(&self) -> bool {
        self.api_key.is_some()
    }

    /// Ask Claude `question` using `context` as the only background.
    /// Returns the answer text plus token usage.
    pub async fn ask(&self, question: &str, context: &str) -> LlmAnswer {
        let prompt = render_prompt(question, context);

        if let Some(key) = self.api_key.clone() {
            match call_anthropic(&key, &self.model, &prompt, self.timeout).await {
                Ok(answer) => return answer,
                Err(e) => {
                    eprintln!("llm: real call failed ({e}); falling back to mock");
                }
            }
        }
        mock_answer(question, context, &prompt)
    }
}

fn render_prompt(question: &str, context: &str) -> String {
    if context.trim().is_empty() {
        format!(
            "You are an assistant. Answer the question concisely (max 1 sentence).\n\
             Question: {question}",
        )
    } else {
        format!(
            "You are an assistant. Use only the provided context.\n\
             Context: {context}\n\
             Question: {question}\n\
             Answer in one short sentence.",
        )
    }
}

fn count_tokens(text: &str) -> u32 {
    let words = text.split_whitespace().count();
    if words == 0 {
        return 0;
    }
    ((words as f32) * 1.3).ceil() as u32
}

fn mock_answer(question: &str, context: &str, prompt: &str) -> LlmAnswer {
    // Deterministic mock — pulls a few key tokens from the context if
    // present, else a generic placeholder. Good enough for the demo /
    // tests.
    let q = question.trim().to_lowercase();
    let ctx = context.trim();
    let text = if ctx.is_empty() {
        format!("(mock) Without context I can't answer '{question}'.")
    } else {
        // Try to surface the question token + some surrounding words.
        let lower = ctx.to_lowercase();
        if let Some(pos) = lower.find(&q) {
            let from = pos.saturating_sub(40);
            let to = (pos + q.len() + 60).min(ctx.len());
            let snippet = &ctx[from..to];
            format!(
                "(mock) Based on the context: …{}… — answer to '{}'.",
                snippet.trim(),
                question,
            )
        } else {
            format!(
                "(mock) The context mentions: {}. Question was: '{}'.",
                first_n_words(ctx, 10),
                question,
            )
        }
    };
    let input_tokens = count_tokens(prompt);
    let output_tokens = count_tokens(&text);
    LlmAnswer {
        text,
        input_tokens,
        output_tokens,
        real: false,
    }
}

fn first_n_words(s: &str, n: usize) -> String {
    s.split_whitespace().take(n).collect::<Vec<_>>().join(" ")
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: [AnthropicMessage<'a>; 1],
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    usage: AnthropicUsage,
}

async fn call_anthropic(
    api_key: &str,
    model: &str,
    prompt: &str,
    timeout: Duration,
) -> Result<LlmAnswer, String> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("client build: {e}"))?;
    let req = AnthropicRequest {
        model,
        max_tokens: MAX_TOKENS,
        messages: [AnthropicMessage {
            role: "user",
            content: prompt,
        }],
    };
    let response = client
        .post(ANTHROPIC_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(&req)
        .send()
        .await
        .map_err(|e| format!("request: {e}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("body: {e}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {body}"));
    }
    let parsed: AnthropicResponse =
        serde_json::from_str(&body).map_err(|e| format!("decode: {e}; body={body}"))?;
    let text = parsed
        .content
        .into_iter()
        .filter(|b| b.block_type == "text")
        .map(|b| b.text)
        .collect::<Vec<_>>()
        .join(" ");
    Ok(LlmAnswer {
        text,
        input_tokens: parsed.usage.input_tokens,
        output_tokens: parsed.usage.output_tokens,
        real: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_returns_deterministic_answer_for_known_context() {
        let client = LlmClient::mock();
        let answer = client
            .ask("rust", "Rust is a systems programming language.")
            .await;
        assert!(!answer.real);
        assert!(answer.text.to_lowercase().contains("rust"));
        assert!(answer.input_tokens > 0);
        assert!(answer.output_tokens > 0);
    }

    #[tokio::test]
    async fn mock_handles_empty_context() {
        let client = LlmClient::mock();
        let answer = client.ask("what is rust", "").await;
        assert!(!answer.real);
        assert!(answer.text.to_lowercase().contains("can't") || answer.text.contains("no context"));
    }

    #[test]
    fn token_counter_matches_eval_heuristic() {
        // Same ceil(words * 1.3) formula used elsewhere in Javis.
        assert_eq!(count_tokens("hello world rust"), 4); // 3 * 1.3 = 3.9 -> 4
        assert_eq!(count_tokens(""), 0);
    }
}
