FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 \
        https://github.com/GaetanBenoitDev/metaMDBG.git /opt/metaMDBG

WORKDIR /opt/metaMDBG

RUN mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j "$(nproc)"


FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        zlib1g \
        libgomp1 \
        minimap2 \
        time \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/metaMDBG/build/bin/metaMDBG /usr/local/bin/

ENTRYPOINT ["metaMDBG"]
