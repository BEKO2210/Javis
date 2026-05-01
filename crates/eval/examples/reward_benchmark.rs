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

use eval::{
    default_reward_corpus, render_reward_markdown, run_reward_benchmark, RewardConfig,
    TeacherForcingConfig,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = parse_arg(&args, "--epochs", 4);
    let seed: u64 = parse_arg(&args, "--seed", 42);
    let reps: u32 = parse_arg(&args, "--reps", 4);
    let only = parse_string(&args, "--only");

    // -- Iter-46 teacher-forcing flags. `--teacher-forcing` switches
    //    the trial schedule from "drive cue+target through R1" to
    //    the proper six-phase cue/delay/prediction/teacher/reward/
    //    tail loop. Every parameter has a sensible default; the
    //    flags are here for parameter sweeps.
    let teacher_on = flag(&args, "--teacher-forcing");
    let teacher_ms: u32 = parse_arg(&args, "--teacher-ms", 40);
    let prediction_ms: u32 = parse_arg(&args, "--prediction-ms", 20);
    let delay_ms: u32 = parse_arg(&args, "--delay-ms", 10);
    let tail_ms: u32 = parse_arg(&args, "--tail-ms", 20);
    let cue_ms: u32 = parse_arg(&args, "--cue-ms", 40);
    let wta_k: usize = parse_arg(&args, "--wta-k", 3);
    let target_clamp: f32 = parse_arg(&args, "--target-clamp-strength", 250.0_f32);
    let reward_positive: f32 = parse_arg(&args, "--reward-positive", 1.0_f32);
    let reward_negative: f32 = parse_arg(&args, "--reward-negative", -0.5_f32);
    let noise_reward: f32 = parse_arg(&args, "--noise-reward", -1.0_f32);
    let homeostasis = flag(&args, "--homeostasis");
    let debug_trials: u32 = if flag(&args, "--debug-trial") { 3 } else { 0 };

    let mut teacher = TeacherForcingConfig {
        enabled: teacher_on,
        cue_ms,
        delay_ms,
        prediction_ms,
        teacher_ms,
        tail_ms,
        target_clamp_strength: target_clamp,
        target_clamp_spike_interval_ms: 0,
        plasticity_during_prediction: false,
        plasticity_during_teacher: true,
        reward_after_teacher: true,
        wta_k,
        negative_reward_for_false_topk: reward_negative,
        positive_reward_for_correct: reward_positive,
        noise_reward,
        homeostatic_normalization: homeostasis,
        debug_trials,
    };

    let corpus = default_reward_corpus();
    eprintln!(
        "Reward benchmark: pairs={} noise_pairs={} vocab={} epochs={epochs} reps={reps} seed={seed} \
         teacher_forcing={teacher_on} wta_k={wta_k} homeostasis={homeostasis}",
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
        // Teacher-forcing is a *training* flag; the baseline arm
        // never uses it regardless of the CLI value.
        let baseline = run_reward_benchmark(
            &corpus,
            &RewardConfig {
                epochs,
                use_reward: false,
                seed,
                reps_per_pair: reps,
                teacher: TeacherForcingConfig::off(),
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
        // For the reward arm, honour the teacher-forcing CLI flag.
        if !teacher.enabled {
            // Defensive: if the user did not pass --teacher-forcing
            // but did pass other teacher-related flags, keep the
            // schedule iter-45-compatible.
            teacher.homeostatic_normalization = false;
        }
        let with_reward = run_reward_benchmark(
            &corpus,
            &RewardConfig {
                epochs,
                use_reward: true,
                seed,
                reps_per_pair: reps,
                teacher,
            },
        );
        eprintln!(
            "  reward-modulated done in {:.1} s",
            t1.elapsed().as_secs_f32(),
        );
        let label = if teacher.enabled {
            "R-STDP + Teacher-Forcing (dopamine + eligibility + clamp)"
        } else {
            "R-STDP (dopamine + eligibility)"
        };
        output.push_str(&render_reward_markdown(label, &with_reward));
    }

    print!("{output}");
}

/// Boolean flag detector — `--name` present → true, absent → false.
fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
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
