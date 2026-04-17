#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/ERR10889718 && pwd)"
RUST_MDBG_DIR="${INPUT_DIR}/rust-mdbg"
OUTPUT_DIR="${RUST_MDBG_DIR}/asf_sample"
mkdir -p "${RUST_MDBG_DIR}" "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# rust-mdbg parameters (must be consistent with any downstream steps)
# ---------------------------------------------------------------------------

# Whether the input is paired-end (R1 + R2 files).
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
    SE_GZ=( "${INPUT_DIR}"/*.fastq.gz )
    SE_PLAIN=( "${INPUT_DIR}"/*.fastq )
    if [[ -f "${SE_GZ[0]:-}" ]]; then
        READS_FILE="${SE_GZ[0]}"
    else
        READS_FILE="${SE_PLAIN[0]}"
    fi
    READS2_FLAG=()
fi

# K: minimizer k-mer order (consecutive minimizers per graph node).
K=7
# L: l-mer length (minimizer alphabet).
L=17
# MINABUND: minimum minimizer occurrence count to retain.
MINABUND=2
# PE_COMBO_DENSITY: fraction of combination hashes retained for PE combo index.
PE_COMBO_DENSITY=0.5
# INTRA_COMBO_DENSITY: fraction of combination hashes retained for intra-read combo index.
INTRA_COMBO_DENSITY=0.05
# COMBO_MAX_DISTANCE: maximum bp distance between R1 and R2 minimizer positions for
# PE combo inclusion. Pairs further apart are discarded. Used during index 
# generation only; not stored in the index.
COMBO_MAX_DISTANCE=1

# rust-mdbg output paths
PREFIX="${RUST_MDBG_DIR}/rust_mdbg_out"
GFA="${PREFIX}.gfa"

# Estimate mean read length, then derive density.
READS_STATS_JSON="${RUST_MDBG_DIR}/reads_stats.json"
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
echo "Using density: ${DENSITY}  (l=${L} / (${MEAN_READ_LEN} * 1.5))"
export RUST_BACKTRACE=full
"$(dirname "$0")/noodle/target/release/noodle" index \
    --reads "${READS_FILE}" \
    "${READS2_FLAG[@]}" \
    --density "${DENSITY}" \
    -l "${L}" \
    --minabund "${MINABUND}" \
    --prefix "${PREFIX}" \
    --pe-combo-density "${PE_COMBO_DENSITY}" \
    --intra-combo-density "${INTRA_COMBO_DENSITY}" \
    --combo-max-distance "${COMBO_MAX_DISTANCE}" \
    --minimizer-batch-size 50000000 \
    --combo-batch-size 50000000 \
    --reads-batch-size 50000000 \
    --reads-shard-size 500000000 \
    --minimizer-shard-size 100000000 \
    --combo-shard-size 100000000 \
2>&1 # | tee "${RUST_MDBG_DIR}/output.txt"

# ---------------------------------------------------------------------------
# asf_sample parameters
# ---------------------------------------------------------------------------

# Number of paths to sample.
N_PATHS=200

# Maximum minimizers per path before termination.
MAX_PATH_MERS=5000

# Minimum reads that must support each extension step.
# Lower values produce longer (but less reliable) paths.
MIN_SUPPORT=2

# Maximum reads examined per k-mer lookup (caps runtime for high-coverage nodes).
MAX_READS_PER_KMER=200

# Random seed for reproducibility.
SEED=42

# Insert size estimation parameters.
# INSERT_SIZE_BINS: comma-separated minimizer-count bin edges.
# After the minimizer-space refactoring, PathResult.distances are always 1 (minimizer
# counts per extension), so bins must be in minimizer-count units, not basepairs.
# At L=17 and typical density for 150bp reads, ~15-20 minimizers/read; a 500bp insert
# corresponds to roughly 50-75 minimizers between read start positions.
INSERT_SIZE_BINS="0,5,10,15,20,30,40,50,75,100,150,200,300,500"
# INSERT_SIZE_INFERENCE: 'nuts' (full MCMC), 'map' (fast), or 'advi'.
INSERT_SIZE_INFERENCE=nuts
# INSERT_SIZE_PATHS: number of sampled paths to use for inference.
INSERT_SIZE_PATHS=50
# MIN_PATH_HASHES_PER_BIN: minimum combo hashes a path-bin must have to be
# included in inference.  Lower values use more data but increase noise.
MIN_PATH_HASHES_PER_BIN=50

OUTPUT="${OUTPUT_DIR}/paths.jsonl"
INSERT_SIZE_JSON="${OUTPUT_DIR}/insert_size.json"
READ_LENGTH_JSON="${OUTPUT_DIR}/read_length.json"

# Read length estimation: number of reads to sample.
READ_LENGTH_READS=1000

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/asf_sample.py" \
    "${PREFIX}" \
    --k "${K}" \
    --n-paths "${N_PATHS}" \
    --max-path-mers "${MAX_PATH_MERS}" \
    --min-support "${MIN_SUPPORT}" \
    --max-reads-per-kmer "${MAX_READS_PER_KMER}" \
    --seed "${SEED}" \
    --output "${OUTPUT}" \
    --estimate-insert-size \
    --insert-size-bins "${INSERT_SIZE_BINS}" \
    --pe-combo-density "${PE_COMBO_DENSITY}" \
    --insert-size-inference "${INSERT_SIZE_INFERENCE}" \
    --insert-size-paths "${INSERT_SIZE_PATHS}" \
    --min-path-hashes-per-bin "${MIN_PATH_HASHES_PER_BIN}" \
    --insert-size-json "${INSERT_SIZE_JSON}" \
    --combo-max-distance "${COMBO_MAX_DISTANCE}" \
    --density "${DENSITY}" \
    --estimate-read-length \
    --read-length-reads "${READ_LENGTH_READS}" \
    --read-length-json "${READ_LENGTH_JSON}" \
    --verbose
