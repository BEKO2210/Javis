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
    default_reward_corpus, default_reward_corpus_v64, render_axis_sweep,
    render_jaccard_floor_diagnosis, render_jaccard_sweep, render_reward_markdown,
    render_target_overlap_sweep, run_axis_sweep, run_determinism_smoke, run_jaccard_bench,
    run_jaccard_floor_diagnosis, run_postmortem_diagnostic, run_reward_benchmark,
    run_target_overlap_arm, ArmMode, C1Config, DgConfig, Iter49Mode, RewardConfig, SweepAxis,
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
    let no_plasticity = flag(&args, "--no-plasticity") || flag(&args, "--frozen-weights");

    // Iter-54: hard-decorrelated R1 → R2 init. Replaces the random
    // FAN_OUT wiring with disjoint-block-per-cue wiring (each vocab
    // word's R1 SDR cells project ONLY into a dedicated R2-E
    // block, shared cells dropped). Mechanical invariant:
    // assert_decorrelated_disjoint ensures pairwise-disjoint R2
    // reach across all cue pairs. iter-54 question: does this
    // unblock the cross-cue specificity gain that iter-53 (random
    // wiring + 16 epochs) failed to produce?
    let decorrelated_init = flag(&args, "--decorrelated-init");

    // Iter-59: R2 neuron count override. `0` (default) keeps the
    // compile-time `R2_N = 2000` baseline (iter-46…58 numerics).
    // Any positive value rebuilds R2 at the requested size.
    let r2_n: u32 = parse_arg(&args, "--r2-n", 0_u32);

    // Iter-62 recall-mode: when set, every plasticity rule is
    // disabled between training and the jaccard-matrix eval phase.
    // Training itself is unchanged. The eval-phase L2 invariant
    // is then re-asserted (pre / post bit-identical).
    let recall_mode_eval =
        flag(&args, "--plasticity-off-during-eval") || flag(&args, "--recall-mode-eval");

    // Iter-60: DG pattern-separation bridge. `--dg-bridge` adds a
    // third region (DG) with per-cue k-of-n hashed SDRs and a
    // sparse mossy-fibre-style projection to R2. The direct
    // R1 → R2 path is gated to `direct_r1r2_weight_scale` (default
    // 0.0 = direct path off, DG is the sole cue-routing layer).
    let dg_bridge = flag(&args, "--dg-bridge");
    let dg_size: u32 = parse_arg(&args, "--dg-size", 4000_u32);
    let dg_k: u32 = parse_arg(&args, "--dg-k", 80_u32);
    let dg_to_r2_fanout: u32 = parse_arg(&args, "--dg-to-r2-fanout", 30_u32);
    let dg_to_r2_weight: f32 = parse_arg(&args, "--dg-to-r2-weight", 1.0_f32);
    let direct_r1r2_weight_scale: f32 = parse_arg(&args, "--direct-r1r2-weight-scale", 0.0_f32);
    let dg_drive_strength: f32 = parse_arg(&args, "--dg-drive-strength", 200.0_f32);

    // Iter-66 (M1): CA1-equivalent C1 readout. `--c1-readout`
    // enables the new layer; `--c1-teacher-strength` sets the
    // M_target neuromodulator pulse during the teacher phase
    // (drives the existing R-STDP rule on the new R2-E → C1
    // synapses, see notes/66-ca1-heteroassoc-readout.md).
    let c1_readout = flag(&args, "--c1-readout");
    let c1_teacher_strength: f32 = parse_arg(&args, "--c1-teacher-strength", 1.0_f32);
    let c1_size: u32 = parse_arg(&args, "--c1-size", 1000_u32);
    let c1_sparsity_k: u32 = parse_arg(&args, "--c1-sparsity-k", 20_u32);
    let c1_from_r2_fanout: u32 = parse_arg(&args, "--c1-from-r2-fanout", 30_u32);
    let c1_init_w_max: f32 = parse_arg(&args, "--c1-init-w-max", 0.5_f32);
    // Iter-66 step 7.5: per-epoch diagnostic logs for the C1 path
    // (R2→C1 weight L2 / Δw stats, teacher-phase C1 spike count,
    // eval-phase C1 spike count, target rank / MRR, raw kWTA ∩
    // canonical-target overlap). Off by default; mandatory for the
    // step 7.5 verdict before the 8-seed step 8 main run.
    let c1_diagnostic = flag(&args, "--c1-diagnostic");
    // Iter-66.5 Path-1 fix (notes/66.5-eval-aligned-c1-rstdp.md):
    // when set together with --c1-readout, the teacher Phase 4
    // omits the canonical R2 target SDR from the clamp so R2 fires
    // its natural cue-driven response. R-STDP on R2-E → C1 then
    // aligns the eval-time R2 cue pattern with the canonical C1
    // target. Default off ⇒ iter-66 behaviour bit-identical (every
    // existing reward_bench snapshot test still passes verbatim).
    let c1_eval_aligned_rstdp = flag(&args, "--c1-eval-aligned-rstdp");
    // Iter-67 BTSP plateau-eligibility on R2-E → C1 (notes/67).
    // Master switch + three locked numeric knobs.  When --c1-btsp
    // is off (default), the iter-66.5 R-STDP path is bit-identical.
    let c1_btsp = flag(&args, "--c1-btsp");
    let c1_btsp_window_ms: f32 = parse_arg(&args, "--c1-btsp-window-ms", 200.0_f32);
    let c1_btsp_strength: f32 = parse_arg(&args, "--c1-btsp-strength", 0.4_f32);
    // Per-post-cell credit-assignment toggle.  Default true (the
    // mechanism the iter-67 ENTRY pre-registers as the binding
    // ingredient).  `--c1-btsp-target-gated` is a positive intent
    // flag (no-op since default is on; documents explicit operator
    // intent in the locked smoke invocation).  `--c1-btsp-no-
    // target-gate` is the ablation that disables per-post-cell
    // locality.  If both are passed, the ablation wins (loud
    // "no" beats quiet "yes").
    let _c1_btsp_explicit_gated = flag(&args, "--c1-btsp-target-gated");
    let c1_btsp_target_gated = !flag(&args, "--c1-btsp-no-target-gate");

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
        recall_mode_eval,
        decorrelated_init,
        r2_n,
        r2_p_connect_override: None,
        dg: DgConfig {
            enabled: dg_bridge,
            size: dg_size,
            k: dg_k,
            to_r2_fanout: dg_to_r2_fanout,
            to_r2_weight: dg_to_r2_weight,
            direct_r1r2_weight_scale,
            drive_strength: dg_drive_strength,
        },
        c1: C1Config {
            enabled: c1_readout,
            size: c1_size,
            sparsity_k: c1_sparsity_k,
            from_r2_fanout: c1_from_r2_fanout,
            init_w_max: c1_init_w_max,
            teacher_strength: c1_teacher_strength,
            diagnostic: c1_diagnostic,
            eval_aligned_rstdp: c1_eval_aligned_rstdp,
            btsp: c1_btsp,
            btsp_window_ms: c1_btsp_window_ms,
            btsp_strength: c1_btsp_strength,
            btsp_target_gated: c1_btsp_target_gated,
        },
    };

    // Iter-58 vocab-scaling stress test: --corpus-vocab 32 (default;
    // iter-46…57 corpus, 16 real + 16 noise pairs ⇒ 32-word vocab) vs
    // --corpus-vocab 64 (iter-58 extension, 32 real + 32 noise pairs
    // ⇒ 64-word vocab). Used to test whether the ~0.20 cross-cue
    // floor scales with vocab (geometric / encoder limit) or stays
    // fixed (architecture / plasticity limit).
    let corpus_vocab: u32 = parse_arg(&args, "--corpus-vocab", 32_u32);
    let corpus = match corpus_vocab {
        32 => default_reward_corpus(),
        64 => default_reward_corpus_v64(),
        other => {
            eprintln!("--corpus-vocab must be 32 or 64 (got '{other}')",);
            std::process::exit(2);
        }
    };

    // Iter-64 mutual-exclusion guard: the bench-mode flags below all
    // claim main() and return; passing more than one is almost
    // certainly a CLI mistake. Earlier iterations relied on
    // first-flag-wins ordering, which silently dropped later flags.
    // Iter-64 fails loudly so the operator sees the conflict.
    let bench_modes_present: Vec<&str> = [
        "--determinism-smoke",
        "--jaccard-bench",
        "--jaccard-floor-diagnosis",
        "--target-overlap-bench",
        "--axis-sweep",
        "--r2-capacity-sweep",
        "--debug-cascade",
        "--c1-readout",
    ]
    .into_iter()
    .filter(|name| flag(&args, name) || parse_string(&args, name).is_some())
    .collect();
    if bench_modes_present.len() > 1 {
        eprintln!(
            "iter-64 mutual-exclusion: pick one bench mode at a time. Got: {:?}",
            bench_modes_present,
        );
        std::process::exit(2);
    }

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
        let seeds_str = parse_string(&args, "--seeds").unwrap_or_else(|| seed.to_string());
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

    // Iter-58 floor-diagnosis branch: run the trained arm at the
    // best-known config across `--seeds`, emit the per-pair Jaccard
    // distribution + top-N high-overlap pairs + per-cue frequency.
    // Used to diagnose whether the ≈0.20 cross-cue floor is a
    // geometric / encoder / dictionary collision artefact (then
    // concentrated on a few SDR-collision pairs) or an architecture
    // / plasticity limit (then uniform across the vocab).
    if flag(&args, "--jaccard-floor-diagnosis") {
        let seeds_str = parse_string(&args, "--seeds").unwrap_or_else(|| seed.to_string());
        let seeds: Vec<u64> = seeds_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        if seeds.is_empty() {
            eprintln!(
                "--jaccard-floor-diagnosis: --seeds must contain at least one parseable u64 (got '{seeds_str}')",
            );
            std::process::exit(2);
        }
        let threshold: f32 = parse_arg(&args, "--floor-threshold", 0.5_f32);
        let top_n: usize = parse_arg(&args, "--floor-top-n", 10_usize);
        eprintln!(
            "[iter-58 floor] seeds={seeds:?} epochs={epochs} reps={reps} \
             teacher_forcing={teacher_on} decorrelated_init={decorrelated_init} \
             corpus_vocab={corpus_vocab} clamp={target_clamp} teacher_ms={teacher_ms} \
             threshold={threshold} top_n={top_n}",
        );
        let cfg = RewardConfig {
            epochs,
            use_reward: true,
            seed,
            reps_per_pair: reps,
            teacher,
        };
        let reports = run_jaccard_floor_diagnosis(&corpus, &cfg, &seeds);
        print!(
            "{}",
            render_jaccard_floor_diagnosis(&reports, threshold, top_n),
        );
        return;
    }

    // Iter-63 cue→target metric on DG-enabled brain. Pre-registered
    // single metric `target_top3_overlap` = mean of iter-44/45's
    // `top3_accuracy` across all epochs (per-epoch decoder top-3 vs
    // canonical target word; iter-51 stable estimator). The earlier
    // ENTRY draft used `prediction_top3_before_teacher` — that was
    // corrected pre-measurement after the positive-control gate
    // caught a metric-definition mismatch (see notes/63-cue-target-
    // metric.md "pre-measurement correction" section). CLI surface
    // deliberately makes mode explicit (--mode <untrained|trained>)
    // so the iter-50-style "wrong-arm-by-accident" bug class is
    // impossible.
    //
    // - `--mode untrained` runs the untrained calibration arm and
    //   prints per-seed + μ + σ + suggested threshold = max(0.05,
    //   μ + 2σ). The user copies the threshold into
    //   `notes/63-cue-target-metric.md` and commits before the
    //   main run.
    // - `--mode trained --threshold T` runs the trained arm, then
    //   re-runs the untrained arm internally with identical seed
    //   list (deterministic given seed; paired-seed invariant
    //   asserted in the renderer), and prints the full paired
    //   sweep + paired t(n−1) + locked branching verdict at
    //   threshold T.
    // - `--mode trained --iter46-baseline` (no --threshold) is
    //   the positive-control path: prints per-seed values + the
    //   iter-63 ENTRY acceptance band [0.07, 0.15] check (iter-51
    //   stable estimator: mean top3_accuracy over 16 epochs = 0.107,
    //   95% CI [0.069, 0.145]). Used to verify the new metric wiring
    //   reproduces iter-51's reading before the calibration step.
    if flag(&args, "--target-overlap-bench") {
        let seeds_str = parse_string(&args, "--seeds").unwrap_or_else(|| seed.to_string());
        let seeds: Vec<u64> = seeds_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        if seeds.is_empty() {
            eprintln!(
                "--target-overlap-bench: --seeds must contain at least one parseable u64 (got '{seeds_str}')",
            );
            std::process::exit(2);
        }
        let mode = match parse_string(&args, "--mode").as_deref() {
            Some("untrained") => ArmMode::Untrained,
            Some("trained") => ArmMode::Trained,
            None => {
                eprintln!(
                    "--target-overlap-bench: --mode <untrained|trained> is \
                     required (no implicit code path; see iter-63 ENTRY note).",
                );
                std::process::exit(2);
            }
            Some(other) => {
                eprintln!(
                    "--target-overlap-bench: --mode must be 'untrained' or 'trained' (got '{other}')",
                );
                std::process::exit(2);
            }
        };

        // For the main-run path (--mode trained, not the iter46-
        // baseline positive control), --threshold must be present
        // *before* any compute starts. Pre-registration discipline:
        // the threshold is locked in notes/63-cue-target-metric.md
        // by the calibration step; main-run invocations that skip
        // the lock should fail loudly without burning a sweep.
        let threshold_main: Option<f32> = if matches!(mode, ArmMode::Trained) && !iter46_baseline {
            let t: f32 = parse_arg(&args, "--threshold", f32::NAN);
            if !t.is_finite() {
                eprintln!(
                    "--target-overlap-bench --mode trained: --threshold <f32> is \
                     required for the main run. Run `--mode untrained` first to \
                     compute μ and σ, lock max(0.05, μ + 2σ) in \
                     notes/63-cue-target-metric.md, commit, then invoke the main \
                     run with that threshold value. (Positive-control path with \
                     --iter46-baseline is exempt and does not need --threshold.)",
                );
                std::process::exit(2);
            }
            Some(t)
        } else {
            None
        };

        let mut cfg = RewardConfig {
            epochs,
            use_reward: true,
            seed,
            reps_per_pair: reps,
            teacher,
        };
        // Mode-specific gates applied here so run_target_overlap_arm's
        // assertions surface configuration mistakes loudly. Iter-63
        // uses `top3_accuracy` (iter-44/45 decoder metric — computed
        // in both teacher and non-teacher schedules), so we do *not*
        // force `cfg.teacher.enabled = true`. The caller chooses the
        // schedule via `--teacher-forcing` exactly as iter-46 / 50 did.
        match mode {
            ArmMode::Untrained => {
                cfg.teacher.no_plasticity = true;
                cfg.use_reward = false;
            }
            ArmMode::Trained => {
                cfg.teacher.no_plasticity = false;
            }
        }

        eprintln!(
            "[iter-63] target_top3_overlap mode={} seeds={seeds:?} epochs={epochs} \
             vocab={} dg={} iter46_baseline={iter46_baseline} \
             decorrelated_init={decorrelated_init} clamp={target_clamp} \
             teacher_ms={teacher_ms} recall_mode_eval={}",
            mode.label(),
            corpus.vocab.len(),
            cfg.teacher.dg.enabled,
            cfg.teacher.recall_mode_eval,
        );

        let arm = run_target_overlap_arm(&corpus, &cfg, &seeds, mode);

        match mode {
            ArmMode::Untrained => {
                let suggested = 0.05_f32.max(arm.mean + 2.0 * arm.std);
                let mut s = String::new();
                s.push_str("### Iter-63 calibration — untrained baseline\n\n");
                s.push_str(&format!(
                    "_n_seeds = {}, vocab = {}, ep = {}, DG = {}, recall-mode = {}_\n\n",
                    seeds.len(),
                    corpus.vocab.len(),
                    epochs,
                    cfg.teacher.dg.enabled,
                    cfg.teacher.recall_mode_eval,
                ));
                s.push_str("| Seed | target_top3_overlap |\n");
                s.push_str("| ---: | ---: |\n");
                for (i, sd) in seeds.iter().enumerate() {
                    s.push_str(&format!("| {} | {:.4} |\n", sd, arm.per_seed[i]));
                }
                s.push('\n');
                s.push_str(&format!(
                    "**Aggregate:** μ = {:.4} ± {:.4} (n = {}).\n\n",
                    arm.mean,
                    arm.std,
                    seeds.len(),
                ));
                s.push_str(&format!(
                    "**Suggested threshold = max(0.05, μ + 2σ) = max(0.05, {:.4}) = {:.4}.**\n\n",
                    arm.mean + 2.0 * arm.std,
                    suggested,
                ));
                s.push_str(
                    "Lock this number in `notes/63-cue-target-metric.md` *before* \
                     invoking `--mode trained --threshold <value>` for the main run.\n",
                );
                print!("{s}");
            }
            ArmMode::Trained => {
                if iter46_baseline {
                    // Positive control path: single-arm acceptance against
                    // the [0.07, 0.15] iter-51 stable-estimator band. No
                    // threshold, no paired sweep — this is a wiring smoke
                    // check. Band corrected pre-measurement (see notes/
                    // 63-cue-target-metric.md "pre-measurement correction"
                    // section): iter-50's ep0 reading was 0.19 but iter-51
                    // showed Arm B oscillates per-epoch with mean = 0.107
                    // and 95% CI [0.069, 0.145] over 16 epochs. The mean
                    // estimator is what `run_target_overlap_arm` returns.
                    let mut s = String::new();
                    s.push_str(
                        "### Iter-63 positive control — iter-46 Arm B baseline through new wiring\n\n",
                    );
                    s.push_str(&format!(
                        "_n_seeds = {}, vocab = {}, ep = {}, iter46_baseline = true._\n\n",
                        seeds.len(),
                        corpus.vocab.len(),
                        epochs,
                    ));
                    s.push_str(
                        "| Seed | target_top3_overlap (mean top3_accuracy) | in [0.07, 0.15]? |\n",
                    );
                    s.push_str("| ---: | ---: | :---: |\n");
                    let mut all_in = true;
                    for (i, sd) in seeds.iter().enumerate() {
                        let v = arm.per_seed[i];
                        let ok = (0.07..=0.15).contains(&v);
                        if !ok {
                            all_in = false;
                        }
                        s.push_str(&format!(
                            "| {} | {:.4} | {} |\n",
                            sd,
                            v,
                            if ok { "✓" } else { "✗" },
                        ));
                    }
                    s.push('\n');
                    s.push_str(&format!(
                        "**Aggregate:** μ = {:.4} ± {:.4}. Acceptance band [0.07, 0.15] \
                         (iter-51 stable estimator: mean top3_accuracy over 16 epochs = \
                         0.107, 95% CI [0.069, 0.145]; iter-46 / iter-50 ep0 reading was \
                         0.19 but per-epoch oscillation makes max-style banding \
                         non-robust).\n\n",
                        arm.mean, arm.std,
                    ));
                    if all_in {
                        s.push_str(
                            "**Verdict: positive control PASSED ✓** — metric wiring \
                             reproduces iter-51's stable estimator within tolerance. \
                             Calibration step may proceed.\n",
                        );
                    } else {
                        s.push_str(
                            "**Verdict: positive control FAILED ✗** — value(s) outside \
                             [0.07, 0.15] band. This is plumbing drift, not 'close \
                             enough'. Fix the wiring before running calibration. **Do \
                             not** pivot architecture (branch B) on a silently-broken \
                             metric pipeline.\n",
                        );
                    }
                    print!("{s}");
                } else {
                    // Main-run path: paired sweep against an internally-
                    // rerun untrained arm at the same seeds. --threshold
                    // was validated up-front so we know it is finite
                    // here.
                    let threshold = threshold_main.expect(
                        "threshold_main was validated to be finite for \
                         --mode trained without --iter46-baseline",
                    );
                    let mut cfg_un = cfg;
                    cfg_un.teacher.no_plasticity = true;
                    cfg_un.use_reward = false;
                    let untrained =
                        run_target_overlap_arm(&corpus, &cfg_un, &seeds, ArmMode::Untrained);
                    print!(
                        "{}",
                        render_target_overlap_sweep(&untrained, &arm, threshold),
                    );
                }
            }
        }
        return;
    }

    // Iter-66 (M1) — CA1-equivalent C1 readout bench mode. Always
    // trained arm (untrained C1 has nothing to read out), always
    // teacher-forcing, always with the C1 layer enabled. Prints
    // per-seed R2 and C1 readouts side-by-side, then the paired
    // (c1_Δ̄ − r2_Δ̄) cross-readout delta. The locked acceptance
    // matrix in notes/66-ca1-heteroassoc-readout.md is applied at
    // the verdict step (iter-66 step 8) — this routing block is
    // the CLI surface; verdict rendering lands with the main run.
    //
    // Always pairs c1_readout=true with run_target_overlap_arm
    // ArmMode::Trained: the iter-66 ENTRY pre-registers the
    // trained arm only.
    if flag(&args, "--c1-readout") {
        let seeds_str = parse_string(&args, "--seeds").unwrap_or_else(|| seed.to_string());
        let seeds: Vec<u64> = seeds_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        if seeds.is_empty() {
            eprintln!(
                "--c1-readout: --seeds must contain at least one parseable u64 (got '{seeds_str}')",
            );
            std::process::exit(2);
        }
        assert!(
            teacher.enabled,
            "iter-66 --c1-readout requires --teacher-forcing (the C1 layer is supervised \
             by the canonical-target SDR clamp during the encoding phase; running without \
             teacher-forcing would skip the M_target gating window entirely).",
        );
        assert!(
            teacher.c1.enabled,
            "internal: --c1-readout flag was set but teacher.c1.enabled was not propagated; \
             check the CLI parser block.",
        );

        let cfg = RewardConfig {
            epochs,
            use_reward: true,
            seed,
            reps_per_pair: reps,
            teacher,
        };

        eprintln!(
            "[iter-66] c1_readout mode=trained seeds={seeds:?} epochs={epochs} \
             vocab={} dg={} c1.size={} c1.sparsity_k={} c1.from_r2_fanout={} \
             c1.teacher_strength={:.2} clamp={target_clamp} teacher_ms={teacher_ms} \
             recall_mode_eval={}",
            corpus.vocab.len(),
            cfg.teacher.dg.enabled,
            cfg.teacher.c1.size,
            cfg.teacher.c1.sparsity_k,
            cfg.teacher.c1.from_r2_fanout,
            cfg.teacher.c1.teacher_strength,
            cfg.teacher.recall_mode_eval,
        );

        let arm = run_target_overlap_arm(&corpus, &cfg, &seeds, ArmMode::Trained);

        // Render the per-seed table for both readouts. The verdict
        // matrix is applied at the iter-66 step-8 main run; this
        // block is the smoke surface.
        let mut s = String::new();
        s.push_str("### Iter-66 — C1 readout (trained arm)\n\n");
        s.push_str(&format!(
            "_n_seeds = {}, vocab = {}, ep = {}, DG = {}, recall-mode = {}, \
             c1.size = {}, c1.teacher_strength = {:.2}_\n\n",
            seeds.len(),
            corpus.vocab.len(),
            epochs,
            cfg.teacher.dg.enabled,
            cfg.teacher.recall_mode_eval,
            cfg.teacher.c1.size,
            cfg.teacher.c1.teacher_strength,
        ));
        s.push_str("| Seed | target_top3_overlap (R2) | c1_target_top3_overlap (C1) | C1 − R2 |\n");
        s.push_str("| ---: | ---: | ---: | ---: |\n");
        for (i, sd) in seeds.iter().enumerate() {
            let r2 = arm.per_seed[i];
            let c1 = arm.c1_per_seed.get(i).copied().unwrap_or(f32::NAN);
            s.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:+.4} |\n",
                sd,
                r2,
                c1,
                c1 - r2,
            ));
        }
        s.push('\n');
        s.push_str(&format!(
            "**Aggregate:** R2 μ = {:.4} ± {:.4}, C1 μ = {:.4} ± {:.4} (n = {}).\n\n",
            arm.mean,
            arm.std,
            arm.c1_mean,
            arm.c1_std,
            seeds.len(),
        ));
        s.push_str(
            "Verdict matrix (locked in notes/66-ca1-heteroassoc-readout.md) is applied \
             at the step-8 main run (8 seeds × 32 epochs). Smoke runs report \
             pipeline integrity only.\n",
        );
        print!("{s}");
        return;
    }

    // Iter-64 mechanism-diagnosis axis sweep. Three isolated axes
    // per notes/64-mechanism-diagnosis.md:
    //   --axis-sweep dg-to-r2-weight        (axis A)
    //   --axis-sweep r2-p-connect           (axis B)
    //   --axis-sweep direct-r1r2-weight-scale (axis C)
    //
    // Two-phase logic: --axis-sweep-phase smoke = 16 epochs (default),
    // full = 32 epochs. Explicit `--epochs N` from the user overrides
    // the phase default.
    //
    // Mutually exclusive with --jaccard-bench / --jaccard-floor-
    // diagnosis / --target-overlap-bench (those return early above);
    // here we additionally fail loudly if any of them was passed
    // alongside --axis-sweep.
    if let Some(axis_arg) = parse_string(&args, "--axis-sweep") {
        // Mutual-exclusion guard. The earlier blocks already returned
        // for their own flags; this catches the case where multiple
        // bench-mode flags are passed simultaneously and we reached
        // the axis-sweep branch via flag-ordering quirks.
        for other in [
            "--jaccard-bench",
            "--jaccard-floor-diagnosis",
            "--target-overlap-bench",
            "--determinism-smoke",
            "--r2-capacity-sweep",
        ] {
            if flag(&args, other) || parse_string(&args, other).is_some() {
                eprintln!(
                    "--axis-sweep: cannot be combined with {other}; pick one bench mode at a time."
                );
                std::process::exit(2);
            }
        }

        let axis = match SweepAxis::parse_cli(&axis_arg) {
            Some(a) => a,
            None => {
                eprintln!(
                    "--axis-sweep: unknown axis '{axis_arg}'. Allowed: dg-to-r2-weight, \
                     r2-p-connect, direct-r1r2-weight-scale."
                );
                std::process::exit(2);
            }
        };

        // Default value lists per axis (notes/64 ENTRY locked).
        let default_values: &[f32] = match axis {
            SweepAxis::DgToR2Weight => &[0.1_f32, 0.5, 1.0, 2.0],
            SweepAxis::R2PConnect => &[0.025_f32, 0.05, 0.10],
            SweepAxis::DirectR1R2WeightScale => &[0.0_f32, 0.1, 0.3, 1.0],
        };
        let values: Vec<f32> = match parse_string(&args, "--values") {
            Some(s) => {
                let parsed: Vec<f32> = s
                    .split(',')
                    .filter_map(|t| t.trim().parse::<f32>().ok())
                    .collect();
                if parsed.is_empty() {
                    eprintln!("--values: must contain at least one parseable f32 (got '{s}')");
                    std::process::exit(2);
                }
                parsed
            }
            None => default_values.to_vec(),
        };

        let seeds_str = parse_string(&args, "--seeds").unwrap_or_else(|| seed.to_string());
        let seeds: Vec<u64> = seeds_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        if seeds.is_empty() {
            eprintln!(
                "--axis-sweep: --seeds must contain at least one parseable u64 (got '{seeds_str}')",
            );
            std::process::exit(2);
        }

        // Two-phase epoch logic. Default = smoke (16). Explicit
        // `--epochs` overrides the phase default.
        let phase_str =
            parse_string(&args, "--axis-sweep-phase").unwrap_or_else(|| "smoke".to_string());
        let phase_default_epochs: usize = match phase_str.as_str() {
            "smoke" => 16,
            "full" => 32,
            other => {
                eprintln!("--axis-sweep-phase: must be 'smoke' or 'full' (got '{other}')");
                std::process::exit(2);
            }
        };
        // Detect whether the user passed --epochs explicitly. parse_arg
        // returns the default 4 silently; we re-check the args for the
        // literal flag to distinguish "user said 4" from "user said
        // nothing".
        let epochs_explicit = args.iter().any(|a| a == "--epochs");
        let epochs_used = if epochs_explicit {
            epochs
        } else {
            phase_default_epochs
        };

        // The `teacher` config built earlier from the CLI flags
        // already carries `decorrelated_init`, `dg.enabled`,
        // `recall_mode_eval`, etc. The axis sweep overrides exactly
        // one parameter per value via `apply_axis_value` inside
        // `run_axis_sweep`; everything else stays as configured.

        eprintln!(
            "[iter-64 axis-sweep] axis={axis} values={values:?} seeds={seeds:?} \
             phase={phase_str} epochs={epochs_used} (explicit={epochs_explicit}) \
             vocab={} dg={dg} decorrelated={dec} recall_mode={rec}",
            corpus.vocab.len(),
            axis = axis.label(),
            dg = teacher.dg.enabled,
            dec = teacher.decorrelated_init,
            rec = teacher.recall_mode_eval,
        );

        let cfg = RewardConfig {
            epochs: epochs_used,
            use_reward: true,
            seed,
            reps_per_pair: reps,
            teacher,
        };
        let result = run_axis_sweep(&corpus, &cfg, &seeds, axis, &values);
        print!("{}", render_axis_sweep(&result));
        return;
    }

    // Iter-59 R2-capacity sweep: loops over `--r2-sizes` (comma-
    // separated list; default 2000,4000) and runs the standard
    // jaccard-bench (trained vs untrained) at each size, printing
    // a single scaling table. Built on top of run_jaccard_bench
    // by overriding `cfg.teacher.r2_n` per iteration. Useful for
    // testing whether the iter-58 vocab=64 floor is a per-cue
    // R2-E block budget limit.
    if flag(&args, "--r2-capacity-sweep") {
        let seeds_str = parse_string(&args, "--seeds").unwrap_or_else(|| seed.to_string());
        let seeds: Vec<u64> = seeds_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        if seeds.is_empty() {
            eprintln!(
                "--r2-capacity-sweep: --seeds must contain at least one parseable u64 (got '{seeds_str}')",
            );
            std::process::exit(2);
        }
        let sizes_str =
            parse_string(&args, "--r2-sizes").unwrap_or_else(|| "2000,4000".to_string());
        let sizes: Vec<u32> = sizes_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        if sizes.is_empty() {
            eprintln!(
                "--r2-capacity-sweep: --r2-sizes must contain at least one parseable u32 (got '{sizes_str}')",
            );
            std::process::exit(2);
        }
        eprintln!(
            "[iter-59] R2 capacity sweep: r2_sizes={sizes:?} seeds={seeds:?} \
             vocab={corpus_vocab} epochs={epochs} reps={reps} \
             teacher_forcing={teacher_on} decorrelated_init={decorrelated_init} \
             clamp={target_clamp} teacher_ms={teacher_ms}",
        );
        // Per-config metrics for the final scaling table.
        struct CapRow {
            r2_n: u32,
            untrained_mean: f32,
            untrained_std: f32,
            trained_mean: f32,
            trained_std: f32,
            delta: f32,
            wallclock_secs: f32,
        }
        let mut rows: Vec<CapRow> = Vec::with_capacity(sizes.len());
        for &r2_n_val in &sizes {
            let mut t = teacher;
            t.r2_n = r2_n_val;
            let cfg = RewardConfig {
                epochs,
                use_reward: true,
                seed,
                reps_per_pair: reps,
                teacher: t,
            };
            let t0 = Instant::now();
            let sweep = run_jaccard_bench(&corpus, &cfg, &seeds);
            let wallclock_secs = t0.elapsed().as_secs_f32();
            let mean_of = |v: &[f32]| -> f32 {
                if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f32>() / v.len() as f32
                }
            };
            let std_of = |v: &[f32], m: f32| -> f32 {
                if v.len() < 2 {
                    return 0.0;
                }
                let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
                var.sqrt()
            };
            let us: Vec<f32> = sweep
                .untrained
                .iter()
                .map(|r| r.jaccard.cross_cue_mean)
                .collect();
            let ts: Vec<f32> = sweep
                .trained
                .iter()
                .map(|r| r.jaccard.cross_cue_mean)
                .collect();
            let um = mean_of(&us);
            let tm = mean_of(&ts);
            let row = CapRow {
                r2_n: r2_n_val,
                untrained_mean: um,
                untrained_std: std_of(&us, um),
                trained_mean: tm,
                trained_std: std_of(&ts, tm),
                delta: tm - um,
                wallclock_secs,
            };
            eprintln!(
                "[iter-59] r2_n={} done | untrained {:.3}±{:.3} | trained {:.3}±{:.3} | Δ {:+.3} | {:.1} s",
                row.r2_n,
                row.untrained_mean,
                row.untrained_std,
                row.trained_mean,
                row.trained_std,
                row.delta,
                row.wallclock_secs,
            );
            rows.push(row);
        }
        // Final scaling table.
        println!("\n## Iter-59: R2 capacity scaling sweep\n");
        println!(
            "_vocab={corpus_vocab}, n_seeds={}, epochs={epochs}, decorrelated_init=true, clamp={target_clamp} nA, teacher_ms={teacher_ms} ms_\n",
            seeds.len(),
        );
        println!(
            "| R2_N | cells/cue (block_size) | Untrained cross | Trained cross | Δ cross | Wallclock |",
        );
        println!("| ---: | ---: | ---: | ---: | ---: | ---: |",);
        for row in &rows {
            // R2-E ≈ R2_N × (1 - R2_INH_FRAC). At iter-46/58 default
            // 0.30, R2-E = 0.7 × R2_N. block_size = R2-E / vocab.
            let r2_e_est = (row.r2_n as f32 * 0.70) as u32;
            let block_size = r2_e_est / corpus_vocab.max(1);
            println!(
                "| {} | {} | {:.3} ± {:.3} | {:.3} ± {:.3} | {:+.3} | {:.0} s |",
                row.r2_n,
                block_size,
                row.untrained_mean,
                row.untrained_std,
                row.trained_mean,
                row.trained_std,
                row.delta,
                row.wallclock_secs,
            );
        }
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
