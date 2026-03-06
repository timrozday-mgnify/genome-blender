#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/cuttlefish3"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/tmp"

# --- Build parameters ---
K=31                    # -k: k-mer length (default: 31)
MIN_LEN=12              # --min-len: minimizer length (default: 12)
CUTOFF=1                # -c: frequency cutoff for (k+1)-mers (default: refs=1, reads=2)
INPUT_MODE="--read"     # --read for FASTQ (sequencing reads), --ref for FASTA (reference seqs)
COLOR=false             # --color: color the compacted graph

# Collect input FASTQ files as --seq arguments.
# Alternatively use --list for a text file of paths, or --dir for a directory.
FASTQ_FILES=("${INPUT_DIR}"/*.fastq)
SEQ_ARGS=""
for f in "${FASTQ_FILES[@]}"; do
    SEQ_ARGS="${SEQ_ARGS} --seq /input/$(basename "${f}")"
done

# Build optional flags
OPT_FLAGS=""
${COLOR} && OPT_FLAGS="${OPT_FLAGS} --color"

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    -v cuttlefish3_tmp:/tmp/cf3 \
    cuttlefish3 build \
    ${INPUT_MODE} \
    ${SEQ_ARGS} \
    -k "${K}" \
    --min-len "${MIN_LEN}" \
    -o /output/cuttlefish3_out \
    -w /tmp/cf3 \
    -c "${CUTOFF}" \
    ${OPT_FLAGS}
