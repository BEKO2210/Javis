# Javis

Brain-inspired knowledge graph, built as a research prototype.

**Goal:** a token-efficient memory layer for LLMs/agents — only the regions
relevant to a query "fire", and only their content is sent to the model.

**Approach:** fully neuromorphic. Spiking neurons (LIF), pair-based STDP
plasticity. Concepts are encoded as cell assemblies, not as embeddings.

## Status

- `crates/snn-core` — LIF neurons, synapses, STDP. 6 tests passing.
- Research log under `notes/`.

## Run

```sh
cargo test
```

CPU-only; data layout is flat (`Vec<f32>`) so a later GPU port via
`wgpu` / `candle` is mechanical.
