#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/bifrost"
mkdir -p "${OUTPUT_DIR}"

# --- Build parameters ---
K=31                    # k-mer length (default: 31)
THREADS=4               # number of threads
BLOOM_BITS=24           # Bloom filter bits per k-mer (default: 24)
CLIP_TIPS=true          # -i: clip tips shorter than k k-mers
DEL_ISOLATED=true       # -d: delete isolated contigs shorter than k k-mers
COLORS=false            # -c: color the compacted de Bruijn graph
FASTA_OUT=false         # -f: output fasta instead of gfa
BFG_OUT=false           # -b: output bfg/bfi format instead of gfa
VERBOSE=false           # -v: print info messages during execution
NO_COMPRESS=false       # -n: do not compress output files
NO_INDEX=false          # -N: do not make index file

# Collect input FASTQ files as -s (sequencing reads) arguments.
# Use -r instead of -s for reference files (all k-mers kept, no filtering).
FASTQ_FILES=("${INPUT_DIR}"/*.fastq)
SEQ_ARGS=""
for f in "${FASTQ_FILES[@]}"; do
    SEQ_ARGS="${SEQ_ARGS} -s /input/$(basename "${f}")"
done

# Build optional flags
OPT_FLAGS=""
${CLIP_TIPS}    && OPT_FLAGS="${OPT_FLAGS} -i"
${DEL_ISOLATED} && OPT_FLAGS="${OPT_FLAGS} -d"
${COLORS}       && OPT_FLAGS="${OPT_FLAGS} -c"
${FASTA_OUT}    && OPT_FLAGS="${OPT_FLAGS} -f"
${BFG_OUT}      && OPT_FLAGS="${OPT_FLAGS} -b"
${VERBOSE}      && OPT_FLAGS="${OPT_FLAGS} -v"
${NO_COMPRESS}  && OPT_FLAGS="${OPT_FLAGS} -n"
${NO_INDEX}     && OPT_FLAGS="${OPT_FLAGS} -N"

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    bifrost build \
    ${SEQ_ARGS} \
    -k "${K}" \
    -B "${BLOOM_BITS}" \
    -o /output/bifrost_out \
    ${OPT_FLAGS} \
    -t "${THREADS}"
