FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 \
        https://github.com/pmelsted/bifrost.git /opt/bifrost

WORKDIR /opt/bifrost

RUN mkdir build && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCOMPILATION_ARCH=OFF \
        .. \
    && make -j "$(nproc)" \
    && make install


FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/Bifrost /usr/local/bin/
COPY --from=builder /usr/local/lib/libbifrost.* /usr/local/lib/
COPY --from=builder /usr/local/include/bifrost/ /usr/local/include/bifrost/

RUN ldconfig

ENTRYPOINT ["Bifrost"]
