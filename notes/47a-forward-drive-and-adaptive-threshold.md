# Iter 47a — Forward-drive scaling + adaptive threshold

## Why iter-47

`notes/46` ended on a measurable but negative diagnosis:
`target_clamp_hit_rate = 1.00` every epoch in the teacher arm, but
`correct_minus_incorrect_margin ∈ [-0.06, -0.03]` — the canonical
target neurons fired *less* than the rest under cue-only recall.
The chain "clamp → R-STDP eligibility → reward → recurrent weights"
was each verifiably alive; what was missing was a fair shot at the
recurrent path against the dominant R1 → R2 forward drive (90–180
active R2 cells per cue vs. 30 canonical target cells).

The iter-47-architecture-decision note (research-backed) prescribed
**iter-47a-2**: drop INTER_WEIGHT (Brunel scaling) and switch on
the iter-44 `IntrinsicParams` adaptive threshold (Diehl & Cook
2015). The mechanism was already implemented since iter-44; iter-47
just had to enable it on R2 with a defensible parameter set.

## What iter-47a ships

### Phase 0 (commit `99540d0`) — implementation

- `INTER_WEIGHT 2.0 → 1.0` (after the sweep below documented why
  not 0.5 or 0.7).
- `enable_intrinsic_plasticity(intrinsic())` on R2 with the
  Diehl-Cook parameter set:
  `alpha_spike = 0.05, tau_adapt_ms = 2000, a_target = 0,
   beta = 1.0, offset_min = 0, offset_max = 5`.
- `TrialOutcome.pred_target_hits` — counts canonical-target
  neurons that fired in the prediction phase.
- 5 new `RewardEpochMetrics` fields:
  * `r2_active_pre_teacher_{mean,p10,p90}` — distribution of
    cue-driven activity, the iter-47 acceptance band `[25, 70]`
    for `mean`.
  * `target_hit_pre_teacher_mean` — direct readout of "did the
    cue alone reach the right cells?".
  * `selectivity_index` — Diehl-Cook normalisation:
    `target_hit / |target| − non_target / (|R2_E| − |target|)`.
    The single cleanest number — `> 0` = targets fire over-
    proportionally; `< 0` = the iter-46 symptom that iter-47
    must flip.
- `render_markdown` emits a second per-arm table for these.
- All 9 eval lib tests stayed green; clippy `-D warnings` clean.

### Phase 1 — 4-epoch smoke sweep (the entire reason for the
sequential protocol)

The acceptance criteria fixed before Phase 1, per Bekos's protocol:

> after 4 epochs: `mean(margin) > 0.0` **AND**
> `p10(r2_active_pre_teacher) ≥ 5` **AND**
> `mean(r2_active_pre_teacher) ∈ [25, 70]` **AND**
> `mean(pred_top3_before_teacher) ≥ 0.10`
> ≥ 3/4 ⇒ proceed to Phase 2.

Three sweep points were measured (each is one full
`reward_benchmark --epochs 4 --reps 4 --teacher-forcing` run on
the same 16-pair + 16-noise corpus, seed 42). All numbers below
are the **last epoch of the smoke run**:

| INTER_WEIGHT | r2_act mean | r2_act p10/p90 | tgt_hit | selectivity | margin | pred-t3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.5** | **0.8** | 0 / 3 | 0.00 | -0.0005 | -0.01 | 0.00 |
| **1.0** | **139** | 88 / 165 | 2.59 | -0.0005 | -0.02 | 0.00 |
| **0.7** | **507**(*) | 8 / **1599** | 9.38 | -0.0047 | -0.02 | 0.06 |

(*) **0.7 is bistable**: epochs 0-2 stabilise around `r2_act ≈ 10`,
epoch 3 explodes to `mean = 507 / p90 = 1599` — a recurrent
cascade ate ≈ 75 % of the R2-E pool in one trial. Adaptive
threshold (`offset_max = 5`) cannot keep up with the exponential
blow-up. This is the failure mode Litwin-Kumar & Doiron (2014)
warned about.

### Phase 1.5 — Acceptance check

| Sweep point | margin > 0 | p10 ≥ 5 | mean ∈ [25, 70] | pred-t3 ≥ 0.10 | total |
| --- | :-: | :-: | :-: | :-: | :-: |
| 0.5 | ✗ | ✗ | ✗ | ✗ | **0/4** |
| 1.0 | ✗ | ✓ | ✗ (above) | ✗ | **1/4** |
| 0.7 | ✗ | ✓ | partial / cascade | ✗ | **1.5/4** |

**No sweep point ≥ 3/4. iter-47a-2 alone does not pass.**

But — and this is what makes the iter much more than a "no" — the
diagnosis is dramatically sharper than iter-46's:

1. The negative-margin symptom **moved with INTER_WEIGHT**.
   At 1.0, `selectivity` rose from -0.022 (epoch 0) to -0.0005
   (epoch 3), and `tgt_hit_mean` grew monotonically 1.16 → 2.59.
   That is the *first time* in iter-46/47 we have seen
   correct-direction motion on the canonical-target metric over
   epochs. The mechanism is right; it just runs out of headroom.
2. The 0.7 bistability **proves a second-order bottleneck exists**:
   even when forward drive is in the right ballpark, recurrent
   STDP-grown weights cascade once they cross a threshold.
   Adaptive θ alone (a per-cell rate-control rule) cannot prevent
   network-wide instability.

## What this rules in / rules out for iter-48

**Ruled out** (by Phase 1 evidence):
- Pfad 1 (INTER_WEIGHT alone) — no sweep point passes acceptance.
- Adaptive θ as the sole sparsity mechanism — fails against
  recurrent cascade.

**Ruled in** (by Phase 1 evidence):
- The forward-vs-recurrent balance is genuinely at the right
  order of magnitude — at INTER_WEIGHT = 1.0 the network is
  noisy but not catastrophically so.
- The Diehl-Cook adaptive threshold mechanism *does work* at
  the per-cell level (target_hit grew monotonically).
- Hard sparsity control (k-WTA, Pfad 47a-3 from the architecture
  note) is **necessary**, not optional. The bistability at
  INTER_WEIGHT = 0.7 makes that explicit.

**Open** (Phase 1 cannot rule on these):
- Whether k-WTA on R2 *during* the simulation (not post-hoc on
  the metric) is enough on top of INTER_WEIGHT = 1.0 + adaptive
  θ to flip the margin sign.
- Whether bounded R-STDP on R1 → R2 (47b) is needed even with
  k-WTA, or only as a downstream-iter optimisation.

## Acceptance criteria — measured

| # | Criterion | Status |
| - | --- | --- |
| 1 | Code compiles | ✅ |
| 2 | Existing tests still green | ✅ (9 lib) |
| 3 | Smoke test runs in reasonable time | ✅ (~1 min per sweep point) |
| 4 | Evaluation does not modify weights | ✅ (unchanged from iter-46) |
| 5 | Teacher phase not counted as recall | ✅ (unchanged from iter-46) |
| 6 | Clamp activates target R2 SDR | ✅ (`clamp_hit_rate = 1.00`) |
| 7 | margin > 0 over epochs | ❌ stays `[-0.04, -0.01]` |
| 8 | top-3 ≥ 0.20 stable | ❌ stays at chance |
| 9 | New iter-47 metrics in place | ✅ (5 fields + percentile helper) |
| 10 | Sweep-evidence-driven decision | ✅ (this note) |

## Where this leaves iter-48

The architecture-decision note's ranked recommendation was
`iter-47a-2 → 47a-3 (k-WTA) → 47b (R-STDP on R1→R2)`. Phase 1
of iter-47a-2 produced enough evidence to bypass the speculative
ordering: 47a-3 is unavoidable (the bistability proves it), and
47b is unanswered until 47a-3 lands.

**Concrete iter-48 entry**: implement per-step k-WTA on R2 spike
delivery (or, equivalently, a stricter recurrent-input cap — e.g.
a hard `total_recurrent_input ≤ k * v_threshold` rule per cell
per step). The iter-47 sparsity metrics (`r2_active_pre_teacher_*`
and `selectivity_index`) are already wired to A/B-test it cleanly.
