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
    default_reward_corpus, render_jaccard_sweep, render_reward_markdown,
    run_determinism_smoke, run_jaccard_bench, run_postmortem_diagnostic,
    run_reward_benchmark, Iter49Mode, RewardConfig, TeacherForcingConfig,
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
    // Iter-48 A/B: keep iSTDP active during the prediction phase
    // (default OFF — preserves iter-46's "evaluation does not
    // modify weights" invariant; STDP and R-STDP stay gated by
    // plasticity_during_prediction either way).
    let istdp_in_prediction = flag(&args, "--istdp-during-prediction");

    // Iter-50 diagnostic: --iter46-baseline reverts INTER_WEIGHT,
    // R2_INH_FRAC, iSTDP params, and skips IntrinsicParams to
    // reproduce the original iter-46 Arm B configuration on the
    // current branch code. Forces iter49_mode = none and ignores
    // teacher_forcing.
    let iter46_baseline = flag(&args, "--iter46-baseline");

    // Iter-52 untrained control: --no-plasticity (alias
    // --frozen-weights) gates every plasticity enable so the brain
    // runs as a pure random-weight forward projection. Forward LIF
    // dynamics, recurrent spike propagation, and the decoder all
    // stay live; only weight updates are silenced. End-of-run L2
    // norm sanity asserts the gate was tight.
    let no_plasticity =
        flag(&args, "--no-plasticity") || flag(&args, "--frozen-weights");

    // Iter-54: hard-decorrelated R1 → R2 init. Replaces the random
    // FAN_OUT wiring with disjoint-block-per-cue wiring (each vocab
    // word's R1 SDR cells project ONLY into a dedicated R2-E
    // block, shared cells dropped). Mechanical invariant:
    // assert_decorrelated_disjoint ensures pairwise-disjoint R2
    // reach across all cue pairs. iter-54 question: does this
    // unblock the cross-cue specificity gain that iter-53 (random
    // wiring + 16 epochs) failed to produce?
    let decorrelated_init = flag(&args, "--decorrelated-init");

    // Iter-49 sweep mode. Three orthogonal interventions on the
    // iter-48 iSTDP collapse mechanism (notes/48-saturation.md):
    //   wmax-cap       — symptom: iSTDP w_max 8.0 → 2.0
    //   a-plus-half    — dynamic: iSTDP a_plus 0.30 → 0.20
    //   activity-gated — temporal: a_plus = 0 first 2 epochs, ramp
    //                    over the next 2, then full
    //   none / absent  — iter-48 baseline (default)
    let iter49_mode = match parse_string(&args, "--iter49-mode").as_deref() {
        Some("wmax-cap") => Iter49Mode::WmaxCap,
        Some("a-plus-half") => Iter49Mode::APlusHalf,
        Some("activity-gated") => Iter49Mode::ActivityGated,
        Some("none") | None => Iter49Mode::None,
        Some(other) => {
            eprintln!(
                "--iter49-mode must be one of: none | wmax-cap | a-plus-half | activity-gated (got '{other}')",
            );
            std::process::exit(2);
        }
    };
    // Iter-47a postmortem mode: trains for `train_epochs` epochs with
    // the configured teacher arm, then runs a single read-only
    // diagnostic trial that captures per-step prediction-phase
    // activity + final v_thresh_offset distribution. Bypasses the
    // normal benchmark output entirely.
    let postmortem = flag(&args, "--debug-cascade");
    let postmortem_train: usize = parse_arg(&args, "--postmortem-train", 4_usize);
    // Iter-46 R1 → R2 gate. Defaults to 1.0 (no gating); pass e.g.
    // 0.3 to halve the forward drive during the prediction phase
    // and let recurrent learning express itself.
    let r1r2_gate: f32 = parse_arg(
        &args,
        "--association-training-gate-r1r2",
        if flag(&args, "--association-training-gate-r1r2") {
            0.3_f32
        } else {
            1.0_f32
        },
    );

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
        r1r2_prediction_gate: r1r2_gate,
        istdp_during_prediction: istdp_in_prediction,
        iter49_mode,
        gated_warmup_epochs: 2,
        gated_ramp_epochs: 2,
        iter46_baseline,
        no_plasticity,
        decorrelated_init,
    };

    let corpus = default_reward_corpus();

    // Iter-53 determinism smoke (Bekos's pre-implementation gate):
    // bypass everything else, run the 1-cue × 3-trial determinism
    // test, print the pairwise Jaccard, exit.
    if flag(&args, "--determinism-smoke") {
        let mut t = teacher;
        t.no_plasticity = true; // forced, see run_determinism_smoke
        let cfg = RewardConfig {
            epochs: 0,
            use_reward: false,
            seed,
            reps_per_pair: 0,
            teacher: t,
        };
        run_determinism_smoke(&corpus, &cfg);
        return;
    }

    // Iter-53 jaccard benchmark: bypass the canonical-hash top-3
    // path (which is forward-drive-confounded per iter-52) and run
    // the decoder-relative same-cue / cross-cue Jaccard sweep. Loops
    // over `--seeds` (comma-separated; defaults to a single `--seed`)
    // and prints both the per-seed table and the aggregate Δ-of-Δ.
    if flag(&args, "--jaccard-bench") {
        let seeds_str = parse_string(&args, "--seeds")
            .unwrap_or_else(|| seed.to_string());
        let seeds: Vec<u64> = seeds_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        if seeds.is_empty() {
            eprintln!(
                "--jaccard-bench: --seeds must contain at least one parseable u64 (got '{seeds_str}')",
            );
            std::process::exit(2);
        }
        eprintln!(
            "[iter-53/54] jaccard sweep: seeds={seeds:?} epochs={epochs} reps={reps} \
             teacher_forcing={teacher_on} iter46_baseline={iter46_baseline} \
             decorrelated_init={decorrelated_init} iter49={}",
            iter49_mode.label(),
        );
        let cfg = RewardConfig {
            epochs,
            use_reward: true,
            seed,
            reps_per_pair: reps,
            teacher,
        };
        let sweep = run_jaccard_bench(&corpus, &cfg, &seeds);
        print!("{}", render_jaccard_sweep(&sweep));
        return;
    }

    // Iter-47a postmortem branch: bypass the normal two-arm output
    // entirely, run a single diagnostic. Always uses the teacher
    // schedule + reward (the only mode the postmortem cares about).
    if postmortem {
        let mut t = teacher;
        t.enabled = true;
        let cfg = RewardConfig {
            epochs: postmortem_train,
            use_reward: true,
            seed,
            reps_per_pair: reps,
            teacher: t,
        };
        let _ = run_postmortem_diagnostic(&corpus, &cfg, postmortem_train);
        return;
    }

    eprintln!(
        "Reward benchmark: pairs={} noise_pairs={} vocab={} epochs={epochs} reps={reps} seed={seed} \
         teacher_forcing={teacher_on} wta_k={wta_k} homeostasis={homeostasis} r1r2_gate={r1r2_gate} \
         istdp_in_pred={istdp_in_prediction} iter49={} iter46_baseline={iter46_baseline} \
         no_plasticity={no_plasticity}",
        corpus.pairs.len(),
        corpus.noise_pairs.len(),
        corpus.vocab.len(),
        iter49_mode.label(),
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
