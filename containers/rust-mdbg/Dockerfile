FROM rust:1.85-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        clang \
        zlib1g-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 \
        https://github.com/ekimb/rust-mdbg.git /opt/rust-mdbg

WORKDIR /opt/rust-mdbg

RUN cargo build --release

RUN git clone --depth 1 \
        https://github.com/lh3/gfatools.git /opt/gfatools \
    && cd /opt/gfatools && make


FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/rust-mdbg/target/release/rust-mdbg /usr/local/bin/
COPY --from=builder /opt/rust-mdbg/target/release/to_basespace /usr/local/bin/
COPY --from=builder /opt/gfatools/gfatools /usr/local/bin/
COPY --from=builder /opt/rust-mdbg/utils/ /usr/local/share/rust-mdbg/utils/

ENV PATH="/usr/local/share/rust-mdbg/utils:${PATH}"

ENTRYPOINT ["rust-mdbg"]
