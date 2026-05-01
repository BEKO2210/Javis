//! Reward-aware pair-association benchmark.
//!
//! Runs the same fixed (cue, target) corpus twice — once with pure
//! STDP only, once with R-STDP + dopamine modulation — and prints a
//! side-by-side per-epoch table. Used to validate the iter-44 reward
//! stack on a task pure correlation cannot solve cleanly (real pairs
//! interleaved with deliberate distractors).
//!
//! ```sh
//! # Quick smoke (4 epochs, ~1 minute on release):
//! cargo run --release -p eval --example reward_benchmark -- --epochs 4
//!
//! # Headline run (8 epochs, ~2 minutes):
//! cargo run --release -p eval --example reward_benchmark -- --epochs 8
//! ```

use std::time::Instant;

use eval::{default_reward_corpus, render_reward_markdown, run_reward_benchmark, RewardConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = parse_arg(&args, "--epochs", 4);
    let seed: u64 = parse_arg(&args, "--seed", 42);
    let reps: u32 = parse_arg(&args, "--reps", 4);
    let only = parse_string(&args, "--only");

    let corpus = default_reward_corpus();
    eprintln!(
        "Reward benchmark: pairs={} noise_pairs={} vocab={} epochs={epochs} reps={reps} seed={seed}",
        corpus.pairs.len(),
        corpus.noise_pairs.len(),
        corpus.vocab.len(),
    );

    let mut output = String::new();
    output.push_str(&format!(
        "## Reward benchmark — pairs {}, noise {}, epochs {epochs}\n\n",
        corpus.pairs.len(),
        corpus.noise_pairs.len(),
    ));

    if only.as_deref() != Some("reward") {
        eprintln!("[1/2] Pure-STDP baseline …");
        let t0 = Instant::now();
        let baseline = run_reward_benchmark(
            &corpus,
            &RewardConfig {
                epochs,
                use_reward: false,
                seed,
                reps_per_pair: reps,
            },
        );
        eprintln!("  baseline done in {:.1} s", t0.elapsed().as_secs_f32());
        output.push_str(&render_reward_markdown(
            "Pure STDP (no neuromodulator)",
            &baseline,
        ));
        output.push('\n');
    }

    if only.as_deref() != Some("baseline") {
        eprintln!("[2/2] Reward-modulated …");
        let t1 = Instant::now();
        let with_reward = run_reward_benchmark(
            &corpus,
            &RewardConfig {
                epochs,
                use_reward: true,
                seed,
                reps_per_pair: reps,
            },
        );
        eprintln!(
            "  reward-modulated done in {:.1} s",
            t1.elapsed().as_secs_f32(),
        );
        output.push_str(&render_reward_markdown(
            "R-STDP (dopamine + eligibility)",
            &with_reward,
        ));
    }

    print!("{output}");
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
