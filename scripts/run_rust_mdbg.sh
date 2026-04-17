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

# noodle index output prefix
PREFIX="${OUTPUT_DIR}/rust_mdbg_out"

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

# COMBO_DENSITY: fraction of combination hashes retained for PE/intra combo indexes.
# Lower values → smaller indexes, faster query, less signal.  0.05 is a good default.
COMBO_DENSITY=0.05

# COMBO_MAX_DISTANCE: maximum bp distance between a R1 and R2 minimizer for PE combo
# inclusion.  Pairs further apart than this are discarded.  Set to roughly 2× the
# expected insert size to keep signal while rejecting spurious long-range pairs.
COMBO_MAX_DISTANCE=2000

# Build LMDB indexes using noodle (replaces the old rust-mdbg binary).
"$(dirname "$0")/noodle/target/release/noodle" index \
    --reads "${READS_FILE}" \
    "${READS2_FLAG[@]}" \
    --density "${DENSITY}" \
    -l "${L}" \
    --minabund "${MINABUND}" \
    --prefix "${PREFIX}" \
    --pe-combo-density "${COMBO_DENSITY}" \
    --intra-combo-density "${COMBO_DENSITY}" \
    --combo-max-distance "${COMBO_MAX_DISTANCE}" \
2>&1 | tee "${OUTPUT_DIR}/output.txt"

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/plot_path_bin_pe_counts.py" \
    --debug-bins "${PREFIX}.path_bins.jsonl" \
    --pe-lmdb "${PREFIX}.pe_index.lmdb" \
    --output "${PREFIX}.path_bin_pe_counts.pdf"

