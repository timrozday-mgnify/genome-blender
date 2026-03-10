FROM perl:5.38-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        libexpat1-dev \
        libdb-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cpanm --notest \
        Module::Install \
        Module::Install::AuthorRequires \
        Module::Install::AutoLicense \
        Module::Install::PodFromEuclid \
        Module::Install::ReadmeFromPod \
        Module::Install::AutoManifest \
        Math::Random::MT \
        Getopt::Euclid \
        BioPerl

RUN git clone \
        git://biogrinder.git.sourceforge.net/gitroot/biogrinder/biogrinder /opt/biogrinder

WORKDIR /opt/biogrinder

# Strip author-only sections from Makefile.PL (auto_doc needs Statistics::R + R
# and pod_from/readme_from extensions; these are not needed for installation).
RUN rm -rf inc \
    && sed -i \
        -e '/^author_requires/d' \
        -e '/^auto_license/d' \
        -e '/^auto_doc/d' \
        -e '/^auto_manifest/d' \
        -e '/^sub auto_doc/,/^}/d' \
        Makefile.PL \
    && perl Makefile.PL && make && make install


FROM perl:5.38-slim

COPY --from=builder /usr/local/lib/perl5/ /usr/local/lib/perl5/
COPY --from=builder /usr/local/bin/grinder /usr/local/bin/
COPY --from=builder /usr/local/bin/average_genome_size /usr/local/bin/
COPY --from=builder /usr/local/bin/change_paired_read_orientation /usr/local/bin/

ENTRYPOINT ["grinder"]
