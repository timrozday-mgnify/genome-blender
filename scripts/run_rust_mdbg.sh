#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/multi_genome_full/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# Whether the input is paired-end (R1 + R2 files).
# Set to 1 to pass R1 and R2 separately via --reads2 (rust-mdbg assigns
# interleaved indices: 1=R1, 2=R2, 3=R1, … enabling arithmetic pairing
# in parse_gfa.py --interleaved-pairs).
# Set to 0 to use a single FASTQ file as single-end input.
PAIRED_END=1

# Detect R1 and R2 files (for paired-end) or collect all FASTQ files (single-end).
if [[ "${PAIRED_END}" -eq 1 ]]; then
    R1=""; for f in "${INPUT_DIR}"/*_R1.fastq.gz "${INPUT_DIR}"/*_R1.fastq; do
        [[ -f "$f" ]] && R1="$f" && break; done
    R2=""; for f in "${INPUT_DIR}"/*_R2.fastq.gz "${INPUT_DIR}"/*_R2.fastq; do
        [[ -f "$f" ]] && R2="$f" && break; done
    if [[ -z "${R1}" || -z "${R2}" ]]; then
        echo "ERROR: could not find R1/R2 FASTQ files in ${INPUT_DIR}" >&2
        exit 1
    fi
    echo "Paired-end reads: R1=${R1}  R2=${R2}"
    READS_FILE="${R1}"
    READS2_FLAG=(--reads2 "${R2}")
else
    # Single-end: prefer .fastq.gz, fall back to .fastq
    SE_GZ=( "${INPUT_DIR}"/*.fastq.gz )
    SE_PLAIN=( "${INPUT_DIR}"/*.fastq )
    if [[ -f "${SE_GZ[0]:-}" ]]; then
        READS_FILE="${SE_GZ[0]}"
    else
        READS_FILE="${SE_PLAIN[0]}"
    fi
    READS2_FLAG=()
fi

# rust-mdbg parameters
# K: minimizer k-mer order (number of consecutive minimizers per graph node).
#   Higher K → more unique nodes, fewer false edges, longer assembly contigs,
#   and less repeat-induced noise in the path combo sketches used for insert
#   size estimation.  K=7 is a good default for typical Illumina data;
#   K=4 is faster but produces more repetitive paths.
K=7

# L: l-mer length (minimizer alphabet).  Together with density (derived below)
#   this controls how many minimizers are sampled per read.
#   Rule of thumb: L ≈ read_length × density × 1.5, so
#     density = L / (read_length × 1.5).
#   L=21 gives ~15–25 minimizers per 150 bp read at the derived density.
#   Increasing L reduces the minimizer density but creates more unique minimizers.
L=21

# MINABUND: minimum number of times a minimizer must appear to be kept in the
#   graph.  Filters low-frequency (likely error-derived) minimizers.
#   MINABUND=2 is standard; increase for very high-coverage data (≥100×).
MINABUND=2

# rust-mdbg output paths
PREFIX="${OUTPUT_DIR}/rust_mdbg_out"
GFA="${PREFIX}.gfa"
MINIMIZER_TABLE="${PREFIX}.minimizer_table"

# Estimate mean read length from the first 1000 reads, then derive density.
# The minimizer length l and density are related by:
#   read_length * density * 1.5 = l  =>  density = l / (read_length * 1.5)
READS_STATS_JSON="${OUTPUT_DIR}/reads_stats.json"
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/reads_summary.py" "${READS_FILE}" -n 1000 \
    --json "${READS_STATS_JSON}"
MEAN_READ_LEN=$(
    /Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python3 -c \
        "import json; d=json.load(open('${READS_STATS_JSON}')); print(int(d['mean'] + 0.5))"
)
DENSITY=$(
    /Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python3 -c \
        "print(f'{${L} / (${MEAN_READ_LEN} * 1.5):.4f}')"
)
echo "Estimated mean read length: ${MEAN_READ_LEN} bp"
echo "Using density: ${DENSITY}  (l=${L} / (${MEAN_READ_LEN} * 0.75))"

# Run rust-mdbg directly (local build from timrozday-mgnify/rust-mdbg, mg-summary branch)
# --dump-read-minimizers  writes the reads LMDB index ({PREFIX}.index.lmdb)
#                         and {PREFIX}.minimizer_table (plain-text TSV: hash <-> l-mer lookup).
#                         When --reads2 is given, R1 gets odd indices (1, 3, 5, …) and R2 gets
#                         even (2, 4, 6, …).
# --dump-kminmer-index    writes the inverted k-min-mer LMDB ({PREFIX}.kminmer_index.lmdb):
#                         for each canonical k-mer of K consecutive minimizers, stores the
#                         sorted list of 1-based read indices that contain it (DUPSORT layout).
# --kminmer-bloom         writes a bloom filter ({PREFIX}.kminmer_bloom.bin) for rapid
#                         membership testing without opening LMDB (~128 MB, ~1% FPR).
# COMBO_DENSITY: fraction of combination hashes retained for PE/intra combo indexes.
# Lower values → smaller indexes, faster query, less signal.  0.05 is a good default.
COMBO_DENSITY=0.05

# COMBO_MAX_DISTANCE: maximum bp distance between a R1 and R2 minimizer for PE combo
# inclusion.  Pairs further apart than this are discarded.  Set to roughly 2× the
# expected insert size to keep signal while rejecting spurious long-range pairs.
COMBO_MAX_DISTANCE=2000

"$(dirname "$0")/../bin/rust-mdbg" \
    "${READS_FILE}" \
    "${READS2_FLAG[@]}" \
    -k "${K}" \
    --density "${DENSITY}" \
    -l "${L}" \
    --minabund "${MINABUND}" \
    --prefix "${PREFIX}" \
    --dump-read-minimizers \
    --dump-kminmer-index \
    --kminmer-bloom \
    --dump-combo-index \
    --combo-density "${COMBO_DENSITY}" \
    --combo-max-distance "${COMBO_MAX_DISTANCE}" \
    --combo-bloom \
2>&1 | tee "${OUTPUT_DIR}/output.txt"

UNIQUE_MINIMIZERS=$(grep -vc '^#' "${MINIMIZER_TABLE}" 2>/dev/null || echo 0)
echo "Unique minimizers in minimizer_table: ${UNIQUE_MINIMIZERS}"

# Build parse_gfa.py paired-end flags.
# --interleaved-pairs: arithmetic pairing (0↔1, 2↔3, …) derived from the
#   interleaved 1-based read indices written by rust-mdbg v2 --reads2 mode.
_PAIRED_FLAGS=()
if [[ "${PAIRED_END}" -eq 1 ]]; then
    _PAIRED_FLAGS+=(--interleaved-pairs)
fi

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/parse_gfa.py" \
    -n 100000 \
    --sample-component-proportion 0.5 \
    --read-minimizers "${PREFIX}" \
    --minimizer-table "${MINIMIZER_TABLE}" \
    --json "${PREFIX}.graph_summary.json" \
    "${_PAIRED_FLAGS[@]}" \
    --combo-k "${K}" \
    --combo-density 0.05 \
    --estimate-insert-size \
    --insert-size-paths 100 \
    --insert-size-bins "0,1000,2000,4000,8000,12000,16000,20000" \
    --insert-size-inference nuts \
    --insert-size-min-bin-hashes 100 \
    --pe-combo-lmdb-out "${PREFIX}.pe_index.lmdb" \
    --intra-combo-lmdb-out "${PREFIX}.intra_index.lmdb" \
    --debug-path-bins "${PREFIX}.path_bins.jsonl" \
    "${GFA}"

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/plot_path_bin_pe_counts.py" \
    --debug-bins "${PREFIX}.path_bins.jsonl" \
    --pe-lmdb "${PREFIX}.pe_index.lmdb" \
    --output "${PREFIX}.path_bin_pe_counts.pdf"

