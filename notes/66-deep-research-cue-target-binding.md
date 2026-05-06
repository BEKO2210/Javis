# Javis iter-66 Deep Research: How to solve Cue→Target Binding after DG Separation

## 1. Problem Summary

Javis has reached a clean separation of three computational concerns and a clean falsification of one of them. The dentate-gyrus (DG) stage produces robust pattern separation: cross-cue Jaccard floor collapses by 16× (iter-60), holding for the vocab64 stimulus set (iter-61, iter-62). The CA3-like recurrent attractor R2 produces stable autoassociative recall: same-cue Jaccard mean = 1.000 across 4/4 seeds, with eval-mode L2 drift exactly zero (iter-62). Determinism and cache integrity hold across 8 seeds (iter-65). The plasticity stack (STDP + iSTDP + BCM + homeostasis + intrinsic-θ + R-STDP + heterosynaptic + structural) does write weight changes — post-training L2 norms differ from pre-training. The system therefore satisfies what Marr (1971) and Treves & Rolls (1994) call the prerequisites for hippocampal-style associative memory: an orthogonalising input stage and a stable autoassociative recurrent stage.

What is precisely missing is *heteroassociative cue→target binding*. The decoder metric `target_top3_overlap` (iter-44/45) measures whether the cue's R2 fingerprint, when read through the per-epoch decoder dictionary, decodes to the canonical-target word in the top three. Iter-63 trained the DG-only configuration over 4 seeds and produced Δ̄ = −0.0027, t(3) = −0.179, n_pass = 0/4 — Branch B FAIL. Iter-64 isolated three axes (`dg_to_r2_weight` ∈ {0.1, 0.5, 1.0, 2.0}, `r2_p_connect` ∈ {0.025, 0.05, 0.10}, `direct_r1r2_weight_scale` ∈ {0.0, 0.1, 0.3, 1.0}) and found a *narrow* operating window only at the iter-46 default for axes A and B, with both extremes producing bit-for-bit deterministic locked states (Δ = 0). Axis C at value 0.3 (perforant-path re-introduction) showed a transient α at smoke + 4-seed full-phase (Δ̄ = +0.0164, t = +0.996). Iter-65's 8-seed robustness check at the same configuration produced per-seed Δ = {+0.0215, +0.0215, +0.0508, −0.0283, −0.0020, +0.0107, −0.0039, −0.0156}, n_pos = 4/8 (chance), Δ̄ = +0.0068, t(7) = +0.779. Per the locked acceptance matrix this is Reject; the 4-seed positive was a sample-frequency artefact of a true ~50 % success-rate distribution.

Further hyperparameter sweeps on the existing architecture are therefore not a sensible next step. Iter-64 already established that all three first-order plasticity-routing axes have either narrow windows at a single value or are locked-state degenerate. The conjunction iter-63 ⊕ iter-65 falsifies the hypothesis that the current STDP-on-recurrent-attractor architecture, irrespective of input gain or connectivity density, writes a systematic cue→target mapping that survives independent seed resampling.

In computational-neuroscience terms, the architecture solves *separation* (DG, Marr 1971; Treves & Rolls 1994; O'Reilly & McClelland 1994) and *completion* (CA3 autoassociation, Treves & Rolls 1994; Rolls 2013), but it does not solve *binding* — the operation that maps a separated cue representation to a separated, non-overlapping target representation. These three operations are computationally distinct (O'Reilly & Rudy 2001; Schapiro et al. 2017). Hebbian/STDP learning in a single recurrent attractor is theoretically suited to autoassociation under sparse coding (Willshaw, Buneman & Longuet-Higgins 1969; Tsodyks & Feigel'man 1988), but heteroassociative cue→target binding requires either (a) a distinct projection from cue-population to target-population whose weights are shaped by a target-conditioned signal, or (b) an explicit binding store, or (c) a three-factor learning rule that gates plasticity by a target-specific eligibility trace. None of these is present in the current Javis stack.

## 2. Literature Findings

### 2.1 Hippocampal DG / CA3 / CA1 models

**Marr, D. (1971). "Simple memory: a theory for archicortex." *Phil. Trans. R. Soc. Lond. B* 262, 23–81.** [doi:10.1098/rstb.1971.0078]
Core claim: The hippocampus is a sparse autoassociative store; the entorhinal–DG–CA3 path is the write/read interface and CA1 acts as the *output decoder* that translates the sparse CA3 code into a neocortically interpretable form. Implies for iter-66: a CA1-equivalent readout layer is canonical, not optional. Supports iter-65 falsification: Marr's model has *two* learned matrices (EC→CA3 and CA3→CA1); Javis only has the CA3 analogue.

**Treves, A., & Rolls, E. T. (1994). "Computational analysis of the role of the hippocampus in memory." *Hippocampus* 4(3), 374–391.** [doi:10.1002/hipo.450040319]
Core claim: CA3 is autoassociative (recurrent collaterals do completion); DG provides separation via mossy fibres which act as a "teaching" input strong enough to impose a new pattern onto CA3; CA3→CA1 is a *heteroassociative* projection that learns to map CA3 attractors back onto the EC-driven cortical representation. Implies for iter-66: the missing operation in Javis is exactly the CA3→CA1 heteroassociative step. Supports iter-65 reading: a single recurrent attractor was never the part of the architecture that did cue→target binding in the canonical model.

**Rolls, E. T. (2013). "The mechanisms for pattern completion and pattern separation in the hippocampus." *Frontiers in Systems Neuroscience* 7, 74.** [doi:10.3389/fnsys.2013.00074]
Core claim: CA3 recurrent collateral synapses use Hebbian (STDP-compatible) learning with sparse representations to perform completion; CA1 then performs heteroassociative recall to neocortex via Schaffer collaterals. Quantifies storage capacity ~0.2–0.3 patterns per CA3 synapse under sparse coding. Implies for iter-66: a CA1-style readout layer with its own Hebbian heteroassociative matrix is the standard mechanism. Supports iter-65 reading.

**O'Reilly, R. C., & McClelland, J. L. (1994). "Hippocampal conjunctive encoding, storage, and recall: avoiding a trade-off." *Hippocampus* 4(6), 661–682.** [doi:10.1002/hipo.450040605]
Core claim: The DG–CA3–CA1 circuit resolves the separation/completion trade-off by *spatially separating* the operations onto distinct subfields with different connectivity statistics. Mossy-fibre detonator synapses impose; perforant-path inputs to CA3 then permit completion; CA1 reads out. Implies for iter-66: do not try to make one structure do both/three operations — the trade-off is real and provably resolved only by structural separation. Supports iter-65 reading directly: trying to make R2 do both completion and binding is the architectural mistake.

**Hasselmo, M. E., Bodelón, C., & Wyble, B. P. (2002). "A proposed function for hippocampal theta rhythm: separate phases of encoding and retrieval enhance reversal of prior learning." *Neural Computation* 14(4), 793–817.** [doi:10.1162/089976602317318965]
Core claim: Theta-phase modulation gates encoding vs retrieval in CA3 and CA1, with cholinergic suppression of recurrent collaterals during encoding to prevent runaway autoassociation contamination of new heteroassociative bindings. Implies for iter-66: if Javis adds a CA1 readout, encoding-vs-retrieval phase gating may be needed to prevent R2 attractor dominance from washing out CA1 weight updates. Neutral on iter-65 falsification (orthogonal mechanism).

**Káli, S., & Dayan, P. (2000). "The involvement of recurrent connections in area CA3 in establishing the properties of place fields: a model." *Journal of Neuroscience* 20(19), 7463–7477.** [doi:10.1523/JNEUROSCI.20-19-07463.2000]
Core claim: CA3 recurrent collaterals shape attractor structure such that CA3 generates *internal* representations that do not correspond directly to external inputs; the actual cue-to-content readout is downstream. Implies for iter-66: a CA3 attractor's internal state is not the right substrate to *read* a target from — even if cue-driven dynamics are stable, the target must be read out by a separate mechanism. Supports iter-65 reading.

**Káli, S., & Dayan, P. (2004). "Off-line replay maintains declarative memories in a model of hippocampal-neocortical interactions." *Nature Neuroscience* 7(3), 286–294.** [doi:10.1038/nn1202]
Core claim: Hippocampal patterns are consolidated to neocortex via replay; the hippocampal store itself is heteroassociative cue→cortical-target mediated by CA3→CA1→EC. Implies for iter-66: replay is a candidate for iter-67+ but the *structural* prerequisite is the heteroassociative readout. Neutral on iter-65 directly.

**Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017). "Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning." *Phil. Trans. R. Soc. B* 372, 20160049.** [doi:10.1098/rstb.2016.0049]
Core claim: Within the hippocampus itself, monosynaptic EC→CA1 supports statistical (heteroassociative) learning while trisynaptic EC→DG→CA3→CA1 supports episodic (autoassociative) learning; these are complementary, not redundant. Implies for iter-66: the *direct EC→CA1* equivalent of `direct_r1r2_weight_scale` (axis C in iter-64) was conceptually right, but it was wired to R2 (CA3 analogue) rather than to a CA1 analogue. Supports iter-65 reading and explains why axis C at 0.3 showed a *transient* signal: it accidentally injected cue information into the same attractor that needed to be read for binding, with no stable mapping target.

### 2.2 Associative memory after pattern separation

**Willshaw, D. J., Buneman, O. P., & Longuet-Higgins, H. C. (1969). "Non-holographic associative memory." *Nature* 222, 960–962.** [doi:10.1038/222960a0]
Core claim: A binary matrix with Hebbian (clipped) updates between a sparse cue layer and a sparse target layer stores up to (ln 2)·N²/(k_cue·k_target) heteroassociations with low error. Implies for iter-66: a Willshaw matrix between a separated cue (DG, k=80 of 4000) and a target population (analogous to Javis target SDR cells) has well-quantified capacity. For Javis vocab64, k=80, N_DG=4000, an N_target ≈ 1000, k_target = 20 Willshaw store can hold thousands of pairs at low recall error. Supports iter-65 reading: the architecture-of-record solution to heteroassociation is a dedicated matrix, not a recurrent attractor.

**Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.** Also: Kanerva, P. (1993). "Sparse distributed memory and related models," in *Associative Neural Memories.*
Core claim: Sparse distributed memory uses a fixed random *address* layer (analogous to DG hashing in Javis) and a learned *content* matrix; cue→target retrieval is a two-stage matrix multiply with thresholding. Capacity scales with the address dimension. Implies for iter-66: Javis's DG hash is *already* an SDM address layer; what is missing is the SDM content matrix from address to target. The architecture is two-thirds of an SDM. Supports iter-65 reading.

**Hopfield, J. J. (1982). "Neural networks and physical systems with emergent collective computational abilities." *PNAS* 79(8), 2554–2558.** [doi:10.1073/pnas.79.8.2554]
Core claim: Recurrent symmetric Hebbian network stores ~0.14·N autoassociative patterns. Implies for iter-66: classical Hopfield is the theoretical analogue of R2's current role; it is purely autoassociative. Supports iter-65 reading: no amount of tuning a Hopfield-style net produces heteroassociation.

**Tsodyks, M. V., & Feigel'man, M. V. (1988). "The enhanced storage capacity in neural networks with low activity level." *Europhysics Letters* 6(2), 101–105.** [doi:10.1209/0295-5075/6/2/002]
Core claim: With sparse coding (activity *f* << 1), capacity of an autoassociative Hebbian net rises to ~1/(f·|ln f|). Implies for iter-66: validates that R2's sparse activity is well-chosen for autoassociation. Does not address heteroassociation. Neutral.

**Krotov, D., & Hopfield, J. J. (2016). "Dense Associative Memory for Pattern Recognition." *NeurIPS 2016.*** [arXiv:1606.01164]
Core claim: Modern Hopfield networks with polynomial/exponential interaction terms have storage capacity exponential in N. They explicitly support cue→target retrieval when the energy function is constructed over (cue, target) concatenated patterns, but they require the target to be part of the stored pattern — not learned through STDP on a recurrent net. Implies for iter-66: a modern Hopfield head over (DG-code, target-SDR) concatenated patterns is a candidate mechanism; capacity is not the bottleneck. Supports iter-65 reading: STDP on recurrent units is not how dense associative memories are written.

**Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need." *ICLR 2021.*** [arXiv:2008.02217]
Core claim: Modern Hopfield networks with the softmax-energy formulation are equivalent to attention; they perform exact heteroassociative retrieval with one update step. Implies for iter-66: a softmax/attention-style readout from DG SDR to a learned target dictionary is mathematically the modern Hopfield retrieval rule and is implementable as a single matrix + softmax. Supports iter-65 reading.

**Palm, G. (1980). "On associative memory." *Biological Cybernetics* 36, 19–31.** [doi:10.1007/BF00337019]
Core claim: Refines Willshaw's analysis; gives optimal information-theoretic capacity for binary heteroassociative matrices at ln 2 ≈ 0.69 bits per synapse. Implies for iter-66: confirms that the storage budget for a Willshaw-style cue→target store in Javis is more than adequate. Supports iter-65 reading.

### 2.3 SNN / STDP paired-association learning

**Cassenaer, S., & Laurent, G. (2007). "Hebbian STDP in mushroom bodies facilitates the synchronous flow of olfactory information in locusts." *Nature* 448, 709–713.** [doi:10.1038/nature05973]
Core claim: STDP in the locust mushroom body shapes Kenyon-cell→β-lobe-neuron synapses; the rule produces stable olfactory representations but the actual *behavioural* paired-association learning (odour→reward) is mediated by a *separate* dopamine-gated reinforcement signal at a downstream layer (Cassenaer & Laurent 2012, *Nature* 482, 47–52, [doi:10.1038/nature10776]). Implies for iter-66: even in the canonical insect paired-association SNN, STDP alone in a recurrent layer does not bind cue to target — a downstream three-factor rule does. Supports iter-65 reading directly.

**Cassenaer, S., & Laurent, G. (2012). "Conditional modulation of spike-timing-dependent plasticity for olfactory learning." *Nature* 482, 47–52.** [doi:10.1038/nature10776]
Core claim: Octopamine (insect reward signal) gates STDP in mushroom-body output neurons; without the third factor, STDP between paired stimuli does not produce conditioned response. Implies for iter-66: a three-factor rule with an explicit target-presence signal at a binding layer is the experimentally established mechanism. Supports iter-65 reading.

**Izhikevich, E. M. (2007). "Solving the distal reward problem through linkage of STDP and dopamine signaling." *Cerebral Cortex* 17(10), 2443–2452.** [doi:10.1093/cercor/bhl152]
Core claim: STDP + eligibility trace + delayed dopamine modulator produces a three-factor rule capable of learning arbitrary input→output mappings under reinforcement. Without the third factor, STDP alone equilibrates to correlation structure, not to task-specified output. Implies for iter-66: this is the canonical three-factor / R-STDP formulation. Javis already has R-STDP in the stack; the question is whether it is *gated by a target-presence signal* at the right synapses. Partially contradicts iter-65 reading — suggests R-STDP could in principle work; consistent with iter-65 if the existing R-STDP is mis-routed (e.g. modulating R2 recurrents rather than a dedicated cue→target projection).

**Frémaux, N., & Gerstner, W. (2016). "Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules." *Frontiers in Neural Circuits* 9, 85.** [doi:10.3389/fncir.2015.00085]
Core claim: Comprehensive review; three-factor rules of the form Δw = η · pre · post · M(t) (where M is a neuromodulator carrying task-relevant signal) are required for goal-directed learning; pure two-factor STDP is task-agnostic. Implies for iter-66: a binding head trained with a three-factor rule, where the modulator signals "target SDR is now active," has theoretical guarantees STDP-alone lacks. Supports iter-65 reading.

**Brea, J., Senn, W., & Pfister, J.-P. (2013). "Matching recall and storage in sequence learning with spiking neural networks." *Journal of Neuroscience* 33(23), 9565–9575.** [doi:10.1523/JNEUROSCI.4098-12.2013]
Core claim: Derives a learning rule for sequence and paired-association recall in SNNs using a teacher-forced KL minimisation; shows pure STDP fails to converge on heteroassociative recall whereas KL-derived three-factor rule succeeds. Implies for iter-66: provides a derivation-grade argument that *some* form of supervised/teacher signal at the target is required. Supports iter-65 reading.

**Bellec, G., Scherr, F., Subramoney, A., et al. (2020). "A solution to the learning dilemma for recurrent networks of spiking neurons." *Nature Communications* 11, 3625.** [doi:10.1038/s41467-020-17236-y]
Core claim: e-prop (eligibility propagation) gives a biologically plausible approximation to BPTT for recurrent SNNs; learns paired-association tasks where pure STDP fails. Implies for iter-66: at the cost of an eligibility trace + a top-down learning signal, recurrent SNNs *can* solve binding. Partially contradicts iter-65 reading: shows the failure is not "recurrent SNN cannot do binding," it is "recurrent SNN with two-factor STDP cannot do binding."

**Diehl, P. U., & Cook, M. (2015). "Unsupervised learning of digit recognition using spike-timing-dependent plasticity." *Frontiers in Computational Neuroscience* 9, 99.** [doi:10.3389/fncom.2015.00099]
Core claim: Pure STDP can learn input-class clusters in a feedforward inhibitory-competitive architecture, but a *labelling step* (post-hoc majority-vote assignment of neurons to classes) is required for readout. Implies for iter-66: even feedforward STDP needs an explicit readout assignment; this is exactly what Javis's decoder dictionary attempts, but the decoder reads R2, not a layer trained against the target. Supports iter-65 reading.

### 2.4 Neuroscience: indexing, complementary learning, three operations

**Teyler, T. J., & DiScenna, P. (1986). "The hippocampal memory indexing theory." *Behavioral Neuroscience* 100(2), 147–154.** [doi:10.1037/0735-7044.100.2.147]
Core claim: The hippocampus stores *indices* (sparse pointers) into neocortical content; recall is the index-driven re-activation of the original cortical pattern via back-projections. Implies for iter-66: a key-value formulation where DG SDR = key and a learned mapping retrieves a target index is the canonical indexing-theory implementation. Supports iter-65 reading.

**Teyler, T. J., & Rudy, J. W. (2007). "The hippocampal indexing theory and episodic memory: updating the index." *Hippocampus* 17(12), 1158–1169.** [doi:10.1002/hipo.20350]
Core claim: Updates indexing theory with 20 years of evidence; index updating uses a modified Hebbian rule at the CA1→subiculum→EC interface, not within CA3 recurrents. Implies for iter-66: the heteroassociative learning happens at the *output* projection of the hippocampus, not within the autoassociative core. Supports iter-65 reading.

**McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). "Why there are complementary learning systems in the hippocampus and neocortex." *Psychological Review* 102(3), 419–457.** [doi:10.1037/0033-295X.102.3.419]
Core claim: Hippocampus uses sparse, separated, fast-binding learning; neocortex uses dense, slow, statistical learning; they are complementary because catastrophic interference would ruin a single shared system. Implies for iter-66: the hippocampal binding mechanism is *fast* and *one-shot*; STDP-stack-on-recurrent-network is *neither*. Supports iter-65 reading.

**O'Reilly, R. C., & Rudy, J. W. (2001). "Conjunctive representations in learning and memory: principles of cortical and hippocampal function." *Psychological Review* 108(2), 311–345.** [doi:10.1037/0033-295X.108.2.311]
Core claim: Separates separation, completion, and conjunctive binding as three distinct computations; argues conjunctive binding requires a layer with appropriate sparsity and Hebbian learning over the *cue-target conjunction*. Implies for iter-66: the binding operation is a third computation, distinct from separation and completion, with its own structural requirements. Directly supports iter-65 reading.

**Norman, K. A., & O'Reilly, R. C. (2003). "Modeling hippocampal and neocortical contributions to recognition memory: a complementary-learning-systems approach." *Psychological Review* 110(4), 611–646.** [doi:10.1037/0033-295X.110.4.611]
Core claim: Concrete computational model with explicit DG (separation), CA3 (completion), CA1 (heteroassociative readout to EC); learning rules differ per layer. CA1 uses error-driven CHL (contrastive Hebbian) for the readout, not pure Hebbian. Implies for iter-66: the canonical hippocampal model uses *contrastive* learning at the CA1 readout, hinting at InfoNCE-style or CHL formulations as biologically grounded engineering choices. Supports iter-65 reading.

### 2.5 Engineering candidates

**Tonegawa, S., Pignatelli, M., Roy, D. S., & Ryan, T. J. (2015). "Memory engram storage and retrieval." *Current Opinion in Neurobiology* 35, 101–109.** [doi:10.1016/j.conb.2015.07.009]
Core claim: Optogenetic engram studies show cue→target binding is implemented by *physical overlap* between cue-engram and target-engram cells in CA1/amygdala/cortex, established by co-activation during encoding. Implies for iter-66: a binding layer where cue and target ensembles *co-activate during encoding* and are linked by Hebbian synapses is biologically attested. Supports iter-65 reading.

**Bittner, K. C., Milstein, A. D., Grienberger, C., Romani, S., & Magee, J. C. (2017). "Behavioral time scale synaptic plasticity underlies CA1 place fields." *Science* 357(6355), 1033–1036.** [doi:10.1126/science.aan3846]
Core claim: CA1 pyramidal cells use *behavioural-timescale synaptic plasticity* (BTSP), a one-shot, plateau-potential-gated rule with a ~seconds eligibility window — categorically not standard STDP. BTSP forms place-field-like associations after one or two trials. Implies for iter-66: the empirically established CA1 binding rule is a wide-window, plateau-gated, one-shot rule, not millisecond STDP. Strongly supports iter-65 reading and points at a specific replacement learning rule for the binding layer.

**Magee, J. C., & Grienberger, C. (2020). "Synaptic plasticity forms and functions." *Annual Review of Neuroscience* 43, 95–117.** [doi:10.1146/annurev-neuro-090919-022842]
Core claim: Review covering STDP, BTSP, heterosynaptic plasticity, and metaplasticity. BTSP at CA1 is the operative rule for episodic-style binding; STDP plays a supporting role in feature shaping. Implies for iter-66: confirms BTSP as the canonical CA1 binding rule. Supports iter-65 reading.

**Schmidgall, S., et al. (2024). "Brain-inspired learning in artificial neural networks: a review." *APL Machine Learning* 2, 021501.** [arXiv:2305.11252]
Core claim: Review of biologically plausible learning rules; identifies that two-factor STDP rarely solves supervised paired-association tasks, while three-factor rules and BTSP-inspired rules close the gap. Implies for iter-66: external corroboration of iter-65 falsification reading from a recent comprehensive review. Supports iter-65 reading.

**Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). "What learning systems do intelligent agents need? Complementary learning systems theory updated." *Trends in Cognitive Sciences* 20(7), 512–534.** [doi:10.1016/j.tics.2016.05.004]
Core claim: Updated CLS theory, with explicit treatment of replay and the role of the hippocampus as a fast key-value store from which content is later consolidated. Implies for iter-66: a key-value store (cue → target index) with replay-based consolidation maps directly onto iter-66/67 staging. Supports iter-65 reading.

**Hawkins, J., & Ahmad, S. (2016). "Why neurons have thousands of synapses, a theory of sequence memory in neocortex." *Frontiers in Neural Circuits* 10, 23.** [doi:10.3389/fncir.2016.00023]
Core claim: Sparse-distributed-representation sequence memory (HTM) implements heteroassociative cue→next-state binding via dendritic-segment-conditioned predictive activations; explicitly *not* a recurrent attractor. Implies for iter-66: another existence-proof that heteroassociative binding in sparse codes is implemented by a dedicated mechanism, not by recurrent autoassociation. Supports iter-65 reading.

### 2.6 Sources potentially contradicting the iter-65 reading

The papers above that *partially contradict* iter-65 are Izhikevich (2007), Bellec et al. (2020), and Brea et al. (2013). All three show that recurrent SNNs *can* learn paired-association tasks, but only under three-factor / e-prop / supervised-derived rules — never under pure two-factor STDP on a recurrent attractor. None of them is a counter-example to "Javis's current architecture cannot do binding"; all of them are existence proofs for "with the right learning rule and a target-conditioned signal, an SNN can." This sharpens rather than challenges the iter-65 reading: the missing element is *target-conditioned plasticity at a binding-dedicated projection*, not more recurrent-attractor tuning.

## 3. Candidate Mechanisms

Five candidate mechanisms ranked by predicted effort-adjusted return.

**M1. CA1-equivalent heteroassociative readout layer (separate matrix)**
A new layer "C1" of N_C1 ≈ 1000 LIF neurons. R2 → C1 is an all-to-all (or sparse) plastic projection trained with a *target-presence* three-factor rule: Δw_{R2→C1} = η · pre_R2 · post_C1 · 1[target SDR is the supervisory pattern at this epoch]. Reads the canonical-target SDR for the decoder. Leans on Marr (1971), Treves & Rolls (1994), Norman & O'Reilly (2003), Schapiro et al. (2017). Fits Javis as an addition: keep R1, DG, R2 unchanged; C1 is the new measurement substrate. Effort: M. Risk: target-presence signal must be wired only at encoding, not retrieval (Hasselmo et al. 2002 phase-gating); without that gate the readout mirrors the cue. Smoke test: 8 seeds × 32 ep, primary metric `c1_target_top3_overlap`, threshold μ_C1_untrained + 2σ_C1_untrained, β if Δ̄ > 0.05 with t > 2.0.

**M2. Willshaw-style binary heteroassociative store DG → Target**
A binary matrix W ∈ {0,1}^{N_DG × N_target}; during encoding, W[i,j] ← 1 if DG cell i and target SDR cell j co-fire on the encoding trial; recall is a thresholded matrix-vector product `target_hat = (DG_code · W) ≥ θ`, with θ set per Willshaw & Palm. Leans on Willshaw, Buneman & Longuet-Higgins (1969), Palm (1980), Kanerva (1988). Fits Javis as a *separate* read path that bypasses R2; R2 stays for autoassociative completion. Effort: S. Risk: ignores spike timing; a binary store may be theoretically clean but depart from the Javis SNN substrate in a way that biases the decoder metric. Smoke test: 8 seeds × 32 ep (training writes W; eval reads), primary metric `willshaw_target_top3_overlap`, threshold 0.20 (Willshaw capacity arithmetic for 64 pairs at the given sparsities predicts ≥ 0.5 at zero noise).

**M3. Modern-Hopfield / softmax-attention readout head**
Store K = (cue_DG_codes) and V = (target_SDRs) as fixed key-value matrices populated during encoding; at recall compute `target_hat = softmax(β · K^T · DG_query) · V`. This is the Ramsauer et al. (2020) modern-Hopfield retrieval. Leans on Krotov & Hopfield (2016), Ramsauer et al. (2020). Fits Javis as a non-spiking readout head over the spiking DG code. Effort: S. Risk: not a learning rule, a memorisation rule — does not exercise the SNN plasticity stack at all and could be argued to side-step rather than answer the research question. Best as an *upper-bound* baseline establishing what the DG code can support information-theoretically. Smoke test: 8 seeds, no training needed; if upper-bound `mhn_target_top3_overlap` < 0.5 the DG code itself is the bottleneck and M1/M4/M5 will also fail.

**M4. Three-factor R-STDP on a dedicated cue→target projection**
Add a direct DG → Target plastic projection (or repurpose `direct_r1r2_weight_scale` axis as DG→Target) trained with R-STDP gated by a target-presence signal: Δw = η · pre · post · M(t), with M(t) = +1 when the target SDR is the supervisory pattern, 0 otherwise. Leans on Izhikevich (2007), Frémaux & Gerstner (2016), Brea et al. (2013), Cassenaer & Laurent (2012). Fits Javis as a new projection plus a small modulator scheduling change. Effort: M. Risk: Javis already has R-STDP in the stack — if it is currently mis-gated, fixing the gating *may* rescue the existing architecture without M1; if the gating is correct and it still fails, this candidate falls through. Smoke test: 8 seeds × 32 ep, primary metric `target_top3_overlap` (existing iter-44/45 metric, comparable to iter-65), threshold per iter-65 acceptance matrix, β if Δ̄ > 0.05, t > 2.0, n_pos ≥ 7/8.

**M5. BTSP-style one-shot plateau-gated binding layer**
A C1-equivalent layer where each target SDR cell, when it fires a "plateau" (modelled as a wide ~1 s eligibility window opened by the supervisory target signal), strengthens all currently active R2 (or DG) afferents within the window. One-shot, gated, asymmetric-window plasticity. Leans on Bittner et al. (2017), Magee & Grienberger (2020). Fits Javis as a new layer with a new plasticity rule (not in the current STDP stack). Effort: L. Risk: requires implementing a new plasticity rule from scratch — no existing Javis rule has the seconds-scale eligibility window or plateau gating. Highest biological fidelity, highest implementation cost. Smoke test: 8 seeds × 32 ep, primary metric `btsp_c1_target_top3_overlap`, threshold 0.10 (BTSP is one-shot so signal should be larger than M1's slow Hebbian).

**M6. Contrastive (CHL / InfoNCE) supervised binding head**
Train R2 → Target projection with contrastive Hebbian or InfoNCE loss: positive pairs = (cue's R2 fingerprint, target SDR), negative pairs = (cue's R2 fingerprint, non-target SDRs). Leans on Norman & O'Reilly (2003) for biological CHL precedent. Fits Javis as a non-spiking optimisation head (or a CHL spiking implementation). Effort: M. Risk: introduces a non-local learning signal that departs further from biological plausibility than M1/M4. Smoke test: 8 seeds × 32 ep, primary metric `chl_target_top3_overlap`, threshold 0.10.

## 4. Recommended Architecture Direction

**Choice: B — CA3/CA1 split, with a new CA1-equivalent layer (mechanism M1) trained heteroassociatively, while R2 retains its autoassociative role.**

Justification, citation-by-citation: Marr (1971), Treves & Rolls (1994), Rolls (2013), and O'Reilly & McClelland (1994) establish that the canonical hippocampal solution to the separation/completion/binding triple is *spatial separation* of the operations onto distinct subfields. Schapiro et al. (2017) and Norman & O'Reilly (2003) make this concrete: CA1 is the heteroassociative readout, with its own learning rule distinct from CA3's. Tonegawa et al. (2015) provides experimental engram-level support. The iter-65 falsification is a textbook instance of the architectural mistake O'Reilly & McClelland (1994) explicitly warned against — making one structure carry both autoassociative completion and heteroassociative binding.

Why this over (A): iter-44/45 decoder is itself a per-epoch readout; if a readout-only fix would work, iter-63→65 would already have shown it. The decoder reads R2 fingerprints that, per iter-65, do not encode cue→target structure. A measurement-only fix is ruled out.

Why over (C, dedicated heteroassociative matrix without a CA1 layer): C is operationally a special case of B with N_C1 = N_target and a degenerate identity layer between them. Building B subsumes C with one additional degree of freedom (a layer with its own neurons and dynamics) at modest extra cost.

Why over (D, Willshaw/SDM): D (M2) and the modern-Hopfield head (M3) are *valuable as baselines* establishing the upper bound the DG code supports. They are not chosen as the primary direction because they bypass the SNN substrate the project exists to study. They should be implemented as *baselines* in iter-66 to bound the metric, but the primary architectural commitment is B.

Why over (E, learning-rule-only change on the existing recurrent attractor): the M4 path. The literature (Izhikevich 2007; Bellec et al. 2020; Brea et al. 2013) shows three-factor rules *can* train recurrent SNNs on paired-association, but at significant complexity (e-prop requires top-down learning signal, Brea requires KL-derived rule). Doing this *on R2* puts the binding signal in conflict with R2's autoassociative role. Doing it on a *new dedicated projection* is M1 in disguise. The cleaner route is B.

What B gives up: (i) parsimony — one more layer to maintain, calibrate, and test; (ii) the chance that R2 alone could be made to bind with the right rule (per Bellec et al. 2020 this is possible in principle); (iii) deferral of BTSP (M5), which has the strongest biological fidelity but the highest implementation cost and is reasonable for iter-67+.

## 5. Minimal iter-66 Proposal

Branch: `iter-66-ca1-heteroassoc-readout`.

Concrete plan: introduce a new layer C1 (N_C1 = 1000 LIF neurons, k=20 sparsity matched to R1 target encoding) with a plastic R2 → C1 projection. Add a target-presence modulator M_target(t) ∈ {0, 1} that is asserted only during encoding epochs when the canonical target SDR is forced onto C1 as a teaching pattern (teacher forcing in the sense of Brea et al. 2013 and Norman & O'Reilly 2003's CHL). The R2→C1 projection learns under three-factor R-STDP gated by M_target, per Frémaux & Gerstner (2016) and Izhikevich (2007). During eval (recall mode, plasticity off, per the iter-62 invariant), M_target is zero and C1's activity is read by a new decoder analogous to the iter-44/45 decoder but trained against C1 fingerprints, not R2.

The single primary metric (pre-registered, no other new metrics): **`c1_target_top3_overlap`** = mean across epochs of top-3 decoder accuracy when decoding C1 (not R2) against the 64-vocab target dictionary. Computed identically to the iter-44/45 procedure but on C1.

Locked seeds: 42, 7, 13, 99, 1, 2, 3, 4 (same 8-seed set as iter-65, for cross-iter comparability).

Locked acceptance matrix (analogous to iter-64; thresholds set so that "α" already exceeds the iter-65 ceiling on R2):
- α (smoke positive): single seed (42) Δ vs untrained-C1 baseline > 0.05 at 32 epochs.
- β (4-seed full phase): seeds {42, 7, 13, 99}, Δ̄ > 0.05, t(3) > 2.0, n_pos = 4/4.
- γ (8-seed robustness): all 8 seeds, Δ̄ > 0.05, t(7) > 2.5, n_pos ≥ 7/8.
- δ (architecture promotion, iter-67 gating): γ holds and `c1_target_top3_overlap` mean > 0.30 (well above random 3/64 ≈ 0.047 and above the iter-63 calibration threshold 0.0621).

Failure-to-accept handling: if α fails, the C1 implementation is broken (debug); if β fails after α passes, the target-presence gating is mis-routed (audit M_target wiring); if γ fails after β passes, document as a second sample-frequency artefact and run M2 (Willshaw) as the upper-bound baseline to determine whether the DG→target *information* is sufficient at all.

Files in the Javis codebase likely to be touched (based on the documented structure):
- `crates/snn-core/`: add `c1.rs` (new layer module); modify the network-construction module that currently wires R1/DG/R2 to add R2→C1 projection and the M_target modulator; extend the plasticity registry to include a target-presence-gated R-STDP rule on the new projection.
- `crates/snn-core/`: extend the recall-mode invariant guard (iter-62) to include C1, ensuring eval-mode L2 on C1 is bit-identical pre/post eval.
- `crates/eval/src/reward_bench.rs`: add C1-decoder construction (per-epoch dictionary on C1 fingerprints) and `c1_target_top3_overlap` computation.
- `crates/eval/examples/reward_benchmark.rs`: add an axis or branch flag enabling C1 readout; preserve backward compatibility with the existing R2 metric for the cross-iter reference column.
- Configuration: a single new scalar `c1_teacher_strength` (controls M_target amplitude during encoding); set to 1.0 for the smoke; do *not* sweep in iter-66 — sweeping is iter-66.5 if β passes.

CLI invocation pattern (analogous to iter-64/65 axis sweeps):

```sh
cargo run --release -p eval --example reward_benchmark -- \
  --branch iter-66-ca1-heteroassoc-readout \
  --readout c1 --c1-teacher 1.0 \
  --seeds 42,7,13,99,1,2,3,4 --epochs 32 \
  --metric c1_target_top3_overlap \
  --acceptance gamma
```

Cross-iter comparability: each run also reports the legacy `target_top3_overlap` on R2 (unchanged metric) so that iter-66 can be compared row-for-row against iter-65 and iter-63.

## 6. Hard Recommendation

**Do this next (exactly one item):**

- Implement candidate M1 (CA1-equivalent C1 layer with target-presence-gated R-STDP on R2→C1, primary metric `c1_target_top3_overlap`) on branch `iter-66-ca1-heteroassoc-readout` per Section 5. This is the minimum-effort, maximum-evidence-supported architectural change consistent with Marr (1971), Treves & Rolls (1994), O'Reilly & McClelland (1994), Norman & O'Reilly (2003), and Schapiro et al. (2017).

**Do not do this yet (literature explicitly suggests these are tempting but premature):**

- Do not sweep new hyperparameters on R2 (iter-64 axes A/B/C are exhausted; iter-65 falsified the most promising axis-C value).
- Do not add a fourth axis to the existing sweep matrix on the assumption an unexplored R2 parameter will reveal a window — O'Reilly & McClelland (1994) argues the trade-off is structural.
- Do not attempt BTSP (M5) before M1 — Bittner et al. (2017) and Magee & Grienberger (2020) make BTSP attractive but it requires a new plasticity rule with seconds-scale eligibility windows that the current Javis stack does not support; biology-vs-effort favours M1 first, BTSP at iter-67 if M1 succeeds and biological-fidelity refinement is the next axis.
- Do not implement consolidation/replay (Káli & Dayan 2004; Kumaran et al. 2016) — replay presupposes a working binding store; Javis does not yet have one.
- Do not change the iter-44/45 R2 decoder — the *decoder* is not the failure point per Section 4 reasoning; changing it conflates measurement and mechanism.

**Kill this path (literature evidence rules out for Javis's current state):**

- Kill "make R2 bind by tuning STDP-stack hyperparameters." Cassenaer & Laurent (2007, 2012), Frémaux & Gerstner (2016), Brea et al. (2013), and Schmidgall et al. (2024) collectively establish that pure two-factor STDP on recurrent units does not solve heteroassociative binding. The iter-65 falsification is consistent with this entire body of work.
- Kill "single-structure does separation+completion+binding." O'Reilly & McClelland (1994) and O'Reilly & Rudy (2001) argue this is structurally precluded.
- Kill "axis C `direct_r1r2_weight_scale` is on the right track." Per Schapiro et al. (2017), the direct EC→CA1 path is biologically real but it lands in *CA1*, not in CA3; landing it in R2 (CA3-equivalent) was the iter-64 architectural confound that produced the iter-65 false positive.

**Keep this as fallback (defer to iter-67+):**

- M2 (Willshaw): implement as a *baseline* in iter-66 to bound `target_top3_overlap` upper limit given the DG code; promote to a primary mechanism only if M1 fails γ but M2 passes its smoke. Willshaw, Buneman & Longuet-Higgins (1969); Palm (1980).
- M3 (modern-Hopfield head): same role as M2, mathematically tighter information-theoretic upper bound. Krotov & Hopfield (2016); Ramsauer et al. (2020).
- M5 (BTSP): the biologically most faithful candidate; deferred only on implementation-cost grounds. Bittner et al. (2017); Magee & Grienberger (2020). Reconsider at iter-67 if M1 passes β but γ is marginal.
- M6 (contrastive / CHL): keep as iter-67+ candidate if M1's three-factor signal is too weak. Norman & O'Reilly (2003).
- Replay/consolidation (Káli & Dayan 2004; Kumaran et al. 2016): iter-68+ once a binding store exists.
- Theta-phase encoding/retrieval gating (Hasselmo et al. 2002): an iter-67+ refinement if M1 binding is contaminated by retrieval-mode crosstalk.
