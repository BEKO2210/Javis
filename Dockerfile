# Multi-stage Dockerfile for the Javis live-visualisation server.
#
# Stage 1 (`builder`): full Rust toolchain pinned to the workspace
# MSRV, builds the `javis-viz` binary in release mode.
#
# Stage 2 (`runtime`): minimal Debian slim with just the binary, the
# frontend assets, and a non-root user. Final image is ~80 MB; most
# of that is the statically-linkable bits of glibc + tls roots that
# `reqwest` needs for the Anthropic API.
#
# Build:
#   docker build -t javis-viz .
# Run:
#   docker run --rm -p 7777:7777 javis-viz
# Health-check:
#   curl localhost:7777/health
#   curl localhost:7777/ready

# ---------- builder ---------------------------------------------------
FROM rust:1.86-bookworm AS builder

WORKDIR /usr/src/javis

# Copy only manifest files first so Cargo can cache the dep graph.
# A change to one .rs file should not invalidate the dep-build layer.
COPY Cargo.toml Cargo.lock ./
COPY crates/snn-core/Cargo.toml      crates/snn-core/Cargo.toml
COPY crates/encoders/Cargo.toml      crates/encoders/Cargo.toml
COPY crates/eval/Cargo.toml          crates/eval/Cargo.toml
COPY crates/llm/Cargo.toml           crates/llm/Cargo.toml
COPY crates/viz/Cargo.toml           crates/viz/Cargo.toml

# Stub-source the entire workspace so `cargo fetch` resolves and
# caches the dep graph without seeing real code yet.
RUN mkdir -p crates/snn-core/src crates/encoders/src crates/eval/src \
             crates/llm/src crates/viz/src \
 && echo 'fn main() {}' > crates/viz/src/main.rs \
 && echo ''            > crates/snn-core/src/lib.rs \
 && echo ''            > crates/encoders/src/lib.rs \
 && echo ''            > crates/eval/src/lib.rs \
 && echo ''            > crates/llm/src/lib.rs \
 && echo ''            > crates/viz/src/lib.rs \
 && cargo fetch --locked

# Real source. Now Cargo only rebuilds the workspace, not the deps.
COPY crates/        crates/

# Touch every workspace lib.rs / main.rs so cargo notices the source
# change and recompiles them after the stub-source layer.
RUN find crates -name '*.rs' -exec touch {} + \
 && cargo build --release --locked --bin javis-viz

# ---------- runtime ---------------------------------------------------
FROM debian:bookworm-slim AS runtime

# CA roots for the Anthropic API; tini for proper PID-1 signal handling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user. The numeric uid is hardcoded so volume mounts on
# the host side can match it without surprises.
RUN groupadd --system --gid 1000 javis \
 && useradd  --system --uid 1000 --gid javis --home /app --shell /usr/sbin/nologin javis

WORKDIR /app

COPY --from=builder /usr/src/javis/target/release/javis-viz /usr/local/bin/javis-viz
COPY --chown=javis:javis crates/viz/static /app/static

USER javis

# Container-friendly defaults. The bind addr must be 0.0.0.0 (not
# loopback) so the host port-forward reaches it; static dir points
# at the layout the COPY above just produced.
ENV JAVIS_BIND_ADDR=0.0.0.0:7777 \
    JAVIS_STATIC_DIR=/app/static \
    RUST_LOG=info \
    JAVIS_LOG_FORMAT=json

EXPOSE 7777

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD curl --fail --silent http://localhost:7777/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/javis-viz"]
