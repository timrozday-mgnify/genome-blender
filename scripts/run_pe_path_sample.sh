#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/multi_genome_deep/output && pwd)"
RUST_MDBG_DIR="${INPUT_DIR}/rust-mdbg"
ASF_SAMPLE_DIR="${RUST_MDBG_DIR}/asf_sample"
OUTPUT_DIR="${RUST_MDBG_DIR}/pe_path_sample"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# rust-mdbg parameters (must match the values used to build the index)
# ---------------------------------------------------------------------------

# K: minimizer k-mer order (consecutive minimizers per graph node).
K=7

# rust-mdbg output prefix (same as in run_asf_sample.sh).
PREFIX="${RUST_MDBG_DIR}/rust_mdbg_out"

# ---------------------------------------------------------------------------
# pe_path_sample parameters
# ---------------------------------------------------------------------------

# Extension mode: pe (connect R1→R2, discard on failure) |
#                 greedy (longest unambiguous path, always emit a result).
SAMPLE_MODE=greedy

# Target number of output paths.
N_PATHS=1000

# Minimum read support per extension step.
# Lower values produce longer paths but may introduce errors.
MIN_SUPPORT=2

# Maximum reads examined per k-mer lookup (caps runtime for high-coverage nodes).
MAX_READS_PER_KMER=200

# In PE mode: abandon a path if the total minimizer count exceeds this without
# connecting to the paired mate.
# In greedy mode: cap applied only while seeking the R2 bridge; lifted once the
# bridge is established (no upper limit after bridging).
MAX_PATH_MERS=1000

# Discard output paths with fewer minimizers than this (0 = no filter).
MIN_PATH_MERS=50

# R1 IDs to draw per desired output path.  Oversampling accounts for reads
# that lack mates in the index or fail to connect.
SAMPLE_FACTOR=10

# Random seed for reproducibility.
SEED=42

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT="${OUTPUT_DIR}/pe_paths.jsonl"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "Running pe_path_sample.py ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/pe_path_sample.py" \
    "${PREFIX}" \
    --output "${OUTPUT}" \
    --k "${K}" \
    --n-paths "${N_PATHS}" \
    --min-support "${MIN_SUPPORT}" \
    --max-reads-per-kmer "${MAX_READS_PER_KMER}" \
    --mode "${SAMPLE_MODE}" \
    --max-path-mers "${MAX_PATH_MERS}" \
    --min-path-mers "${MIN_PATH_MERS}" \
    --sample-factor "${SAMPLE_FACTOR}" \
    --seed "${SEED}" \
    --verbose \
2>&1 | tee "${OUTPUT_DIR}/pe_path_sample.log"

echo "Output written to: ${OUTPUT}"

# ---------------------------------------------------------------------------
# Detect paired-end FASTQ files
# ---------------------------------------------------------------------------

R1=""; for f in "${INPUT_DIR}"/*_R1.fastq.gz "${INPUT_DIR}"/*_R1.fastq; do
    [[ -f "$f" ]] && R1="$f" && break; done
R2=""; for f in "${INPUT_DIR}"/*_R2.fastq.gz "${INPUT_DIR}"/*_R2.fastq; do
    [[ -f "$f" ]] && R2="$f" && break; done
if [[ -z "${R1}" || -z "${R2}" ]]; then
    echo "ERROR: could not find R1/R2 FASTQ files in ${INPUT_DIR}" >&2
    exit 1
fi
echo "Paired-end reads: R1=${R1}  R2=${R2}"

# ---------------------------------------------------------------------------
# reconstruct_sequences parameters
# ---------------------------------------------------------------------------

# Sequence selection mode: best | random | common | consensus
MODE=best

# Minimum chained minimizer anchors for a read alignment to be considered.
MIN_ANCHORS=3

# Minimum anchors for partial reads at path ends (lower recovers end coverage).
TERMINAL_MIN_ANCHORS=1

# Maximum reads examined per minimizer lookup (caps runtime).
MAX_READS_PER_MINIMIZER=200

# Minimum fraction of path minimizers with read coverage.
MIN_COV_FRACTION=0.5

# Scoring components: 1=enable, 0=disable.
WEIGHT_COVERAGE=1   # --weight-coverage / --no-weight-coverage
WEIGHT_MATE=1       # --weight-mate     / --no-weight-mate

# Random seed for reproducibility.
SEED=42

# Gap-fill character for path spans with no read coverage.
GAP_FILL_CHAR=N

# Minimizer density (minimizers per bp); must match rust-mdbg indexing.
# Also used by fit_insert_size.py for Poisson deconvolution.
DENSITY=0.01

RECONSTRUCTED_FA="${OUTPUT_DIR}/pe_paths_reconstructed.fa"

# ---------------------------------------------------------------------------
# Step 2: Reconstruct basepair sequences from sampled paths
# ---------------------------------------------------------------------------

echo "Running reconstruct_sequences.py ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/reconstruct_sequences.py" \
    "${OUTPUT}" \
    --prefix "${PREFIX}" \
    --reads "${R1}" \
    --reads "${R2}" \
    --no-weight-insert \
    --mode "${MODE}" \
    --min-anchors "${MIN_ANCHORS}" \
    --terminal-min-anchors "${TERMINAL_MIN_ANCHORS}" \
    --max-reads-per-minimizer "${MAX_READS_PER_MINIMIZER}" \
    --min-coverage-fraction "${MIN_COV_FRACTION}" \
    $([[ "${WEIGHT_COVERAGE}" -eq 1 ]] && echo "--weight-coverage" || echo "--no-weight-coverage") \
    $([[ "${WEIGHT_MATE}" -eq 1 ]] && echo "--weight-mate" || echo "--no-weight-mate") \
    --seed "${SEED}" \
    --gap-fill-char "${GAP_FILL_CHAR}" \
    --density "${DENSITY}" \
    --output "${RECONSTRUCTED_FA}" \
2>&1 | tee "${OUTPUT_DIR}/reconstruct_sequences.log"

echo "Reconstructed sequences written to: ${RECONSTRUCTED_FA}"

# ---------------------------------------------------------------------------
# fit_insert_size parameters
# ---------------------------------------------------------------------------

INSERT_SIZE_JSON="${OUTPUT_DIR}/insert_size.json"

# ---------------------------------------------------------------------------
# Step 3: Fit insert-size model from minimizer paths and reconstructed sequences
# ---------------------------------------------------------------------------

echo "Running fit_insert_size.py ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/fit_insert_size.py" \
    "${OUTPUT}" \
    --reconstructed "${RECONSTRUCTED_FA}" \
    --density "${DENSITY}" \
    --output "${INSERT_SIZE_JSON}" \
    --verbose \
2>&1 | tee "${OUTPUT_DIR}/fit_insert_size.log"

echo "Insert-size estimates written to: ${INSERT_SIZE_JSON}"
