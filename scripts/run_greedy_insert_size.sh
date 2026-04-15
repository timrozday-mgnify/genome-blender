#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/multi_genome_deep/output && pwd)"
RUST_MDBG_DIR="${INPUT_DIR}/rust-mdbg"
OUTPUT_DIR="${RUST_MDBG_DIR}/greedy_insert_size"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# rust-mdbg parameters (must match the values used to build the index)
# ---------------------------------------------------------------------------

# K: minimizer k-mer order (consecutive minimizers per graph node).
K=7

# rust-mdbg output prefix.
PREFIX="${RUST_MDBG_DIR}/rust_mdbg_out"

# ---------------------------------------------------------------------------
# pe_path_sample parameters
# ---------------------------------------------------------------------------

# Extension mode: greedy — extend as far as possible in both directions.
SAMPLE_MODE=greedy

# Target number of output paths.
N_PATHS=1000

# Minimum read support per extension step.
MIN_SUPPORT=2

# Maximum reads examined per k-mer lookup (caps runtime for high-coverage nodes).
MAX_READS_PER_KMER=200

# In greedy mode: max-path-mers caps only the bridge-seeking phase (forward
# extension before R2 is found).  No limit applies after bridging.
MAX_PATH_MERS=500

# Minimum path length in minimizers; short paths are discarded.
MIN_PATH_MERS=100

# R1 IDs to draw per desired output path.
SAMPLE_FACTOR=10

# Random seed for reproducibility.
SEED=42

# ---------------------------------------------------------------------------
# Output file from step 1
# ---------------------------------------------------------------------------

PATHS_JSONL="${OUTPUT_DIR}/pe_paths.jsonl"

# ---------------------------------------------------------------------------
# Step 1: Greedy path sampling
# ---------------------------------------------------------------------------

echo "Running pe_path_sample.py (greedy, min ${MIN_PATH_MERS} minimizers) ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/pe_path_sample.py" \
    "${PREFIX}" \
    --output "${PATHS_JSONL}" \
    --k "${K}" \
    --mode "${SAMPLE_MODE}" \
    --n-paths "${N_PATHS}" \
    --min-support "${MIN_SUPPORT}" \
    --max-reads-per-kmer "${MAX_READS_PER_KMER}" \
    --max-path-mers "${MAX_PATH_MERS}" \
    --min-path-mers "${MIN_PATH_MERS}" \
    --sample-factor "${SAMPLE_FACTOR}" \
    --seed "${SEED}" \
    --verbose \
2>&1 | tee "${OUTPUT_DIR}/pe_path_sample.log"

echo "Paths written to: ${PATHS_JSONL}"

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

# Minimum anchors for partial reads at path ends.
TERMINAL_MIN_ANCHORS=1

# Maximum reads examined per minimizer lookup (caps runtime).
MAX_READS_PER_MINIMIZER=200

# Minimum fraction of path minimizers with read coverage.
MIN_COV_FRACTION=0.5

# Scoring components: 1=enable, 0=disable.
WEIGHT_COVERAGE=1   # --weight-coverage / --no-weight-coverage
WEIGHT_MATE=1       # --weight-mate     / --no-weight-mate

# Gap-fill character for path spans with no read coverage.
GAP_FILL_CHAR=N

# Minimizer density (minimizers per bp); must match rust-mdbg indexing.
# Also used by estimate_insert_size_pe_combo.py for unit conversion.
DENSITY=0.01

RECONSTRUCTED_FA="${OUTPUT_DIR}/pe_paths_reconstructed.fa"

# ---------------------------------------------------------------------------
# Step 2: Reconstruct basepair sequences from sampled paths
# ---------------------------------------------------------------------------

echo "Running reconstruct_sequences.py ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/reconstruct_sequences.py" \
    "${PATHS_JSONL}" \
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
# estimate_insert_size_pe_combo parameters
# ---------------------------------------------------------------------------

# Inference backend: map (fast MAP point estimate) | nuts (full posterior)
INFERENCE=map

# Comma-separated bin edges (minimizer-count thresholds).
# Must match the values used when building the PE-combo index with rust-mdbg.
INSERT_SIZE_BINS="0,200,400,600,800,1000,1500,2000,3000,4000,6000,8000,12000,16000,20000"

# PE combo thinning density; must match rust-mdbg --pe-combo-density.
PE_COMBO_DENSITY=0.05

# Estimated read length in basepairs; converted to minimizer units internally.
READ_LENGTH=150

# Maximum number of paths used for estimation (random subsample if more).
INSERT_SIZE_PATHS=500

# Minimum combo hashes per path-bin to include in inference.
MIN_PATH_HASHES=50

INSERT_SIZE_JSON="${OUTPUT_DIR}/insert_size.json"

# ---------------------------------------------------------------------------
# Step 3: PE-combo insert size estimation (minimizer-space + bp-space)
# ---------------------------------------------------------------------------

echo "Running estimate_insert_size_pe_combo.py ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/estimate_insert_size_pe_combo.py" \
    "${PREFIX}" \
    "${PATHS_JSONL}" \
    --density "${DENSITY}" \
    --inference "${INFERENCE}" \
    --insert-size-bins "${INSERT_SIZE_BINS}" \
    --pe-combo-density "${PE_COMBO_DENSITY}" \
    --read-length "${READ_LENGTH}" \
    --insert-size-paths "${INSERT_SIZE_PATHS}" \
    --min-path-hashes "${MIN_PATH_HASHES}" \
    --seed "${SEED}" \
    --output "${INSERT_SIZE_JSON}" \
    --verbose \
2>&1 | tee "${OUTPUT_DIR}/estimate_insert_size.log"

echo "Insert-size estimates written to: ${INSERT_SIZE_JSON}"
