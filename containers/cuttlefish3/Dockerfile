FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        curl \
        zlib1g-dev \
        liblz4-dev \
        libbz2-dev \
        libisal-dev \
        nasm \
        pkg-config \
        autoconf \
        automake \
        libtool \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch cuttlefish3 --single-branch --depth 1 \
        https://github.com/COMBINE-lab/cuttlefish.git /opt/cuttlefish

WORKDIR /opt/cuttlefish

RUN mkdir build && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/cuttlefish \
        .. \
    && make -j "$(nproc)" install


FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        zlib1g \
        liblz4-1 \
        libbz2-1.0 \
        libisal2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/cuttlefish/bin/ /usr/local/bin/

ENTRYPOINT ["cuttlefish"]
