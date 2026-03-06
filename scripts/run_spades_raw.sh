#!/usr/bin/env bash
set -euo pipefail

# Run SPAdes with graph simplification disabled (no tip trimming, bubble
# popping, or erroneous-connection removal) to produce a raw de Bruijn graph.
# Uses the hidden --configs-dir flag to supply a modified simplification.info.

INPUT_DIR="$(cd ../genome-blender_run/single_genome_full/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/spades_raw"
mkdir -p "${OUTPUT_DIR}"

# Assembly parameters
K_LIST="31"  # "21,31,41,51"    # comma-separated odd k-mer sizes (range 15-255)

# --- Extract and patch SPAdes configs ---
CONFIGS_DIR="${OUTPUT_DIR}/custom_configs"
if [ ! -d "${CONFIGS_DIR}" ]; then
    docker run --rm \
        -v "${OUTPUT_DIR}:/output" \
        --entrypoint bash \
        quay.io/biocontainers/spades:4.2.0--h8d6e82b_2 \
        -c "cp -r /usr/local/share/spades/configs /output/custom_configs"

    # Replace simplification.info with a version that disables all
    # simplification steps (tip clipping, bulge removal, erroneous-connection
    # removal, isolated-edge removal).
    cat > "${CONFIGS_DIR}/debruijn/simplification.info" << 'SIMP'
; simplification -- all steps disabled

simp
{
    init_clean
    {
        self_conj_condition "{ ec_lb 100, cb 1.0 }"
        early_it_only   false
        activation_cov  -1.

        ier
        {
            enabled                     false
            use_rl_for_max_length       false
            use_rl_for_max_length_any_cov false
            max_length                  0
            max_coverage                0
            max_length_any_cov          0
            rl_threshold_increase       0
        }

        tip_condition   "{ tc_lb 0. }"
        ec_condition    "{ ec_lb 0, cb 0. }"

        disconnect_flank_cov    -1.0
    }

    cycle_iter_count 0

    tc
    {
        condition               "{ tc_lb 0. }"
    }

    br
    {
        enabled             false
        main_iteration_only false
        max_bulge_length_coefficient    3.
        max_additive_length_coefficient 100
        max_coverage            1000.0
        max_relative_coverage       1.1
        max_delta           3
        max_relative_delta      0.1
        max_number_edges        1000
        dijkstra_vertex_limit   3000
        parallel true
        buff_size 10000
        buff_cov_diff 2.
        buff_cov_rel_diff 0.2
        min_identity 0.0
    }

    ec
    {
        condition               "{ ec_lb 0, cb 0. }"
    }

    dead_end {
        enabled false
        condition "{ tc_lb 0. }"
    }

    rcec
    {
        enabled false
        rcec_lb 30
        rcec_cb 0.5
    }

    rcc
    {
        enabled false
        coverage_gap    5.
        max_length_coeff    2.0
        max_length_with_tips_coeff   3.0
        max_vertex_cnt      30
        max_ec_length_coefficient   30
        max_coverage_coeff  2.0
    }

    red
    {
        enabled false
        diff_mult  20.
        edge_sum   10000
        unconditional_diff_mult 0.
    }

    final_tc
    {
        condition               "{ tc_lb 0. }"
    }

    final_br
    {
        enabled             false
        main_iteration_only false
        max_bulge_length_coefficient    3.
        max_additive_length_coefficient 100
        max_coverage            1000.0
        max_relative_coverage       1.1
        max_delta           3
        max_relative_delta      0.1
        max_number_edges        1000
        dijkstra_vertex_limit   3000
        parallel true
        buff_size 10000
        buff_cov_diff 2.
        buff_cov_rel_diff 0.2
        min_identity 0.0
    }

    subspecies_br
    {
        enabled             false
        main_iteration_only false
        max_bulge_length_coefficient    3.
        max_additive_length_coefficient 100
        max_coverage            1000.0
        max_relative_coverage       1.1
        max_delta           3
        max_relative_delta      0.1
        max_number_edges        1000
        dijkstra_vertex_limit   3000
        parallel true
        buff_size 10000
        buff_cov_diff 2.
        buff_cov_rel_diff 0.2
        min_identity 0.0
    }

    complex_tc
    {
        enabled               false
        max_relative_coverage -1
        max_edge_len          100
        condition             "{ tc_lb 0. }"
    }

    cbr
    {
        enabled false
        max_relative_length 5.
        max_length_difference   5
    }

    ier
    {
        enabled                     false
        use_rl_for_max_length       false
        use_rl_for_max_length_any_cov false
        max_length                  0
        max_coverage                0
        max_length_any_cov          0
        rl_threshold_increase       0
    }

    her
    {
        enabled                     false
        uniqueness_length           1500
        unreliability_threshold     4
        relative_threshold          5
    }

    topology_simplif_enabled false

    tec
    {
        max_ec_length_coefficient   55
        uniqueness_length       1500
        plausibility_length     200
    }

    trec
    {
        max_ec_length_coefficient   100
        uniqueness_length       1500
        unreliable_coverage     2.5
    }

    isec
    {
        max_ec_length_coefficient   100
        uniqueness_length       1500
        span_distance       15000
    }

    mfec
    {
        enabled false
        max_ec_length_coefficient   30
        uniqueness_length       1500
        plausibility_length     200
    }

    ttc
    {
        length_coeff    3.5
        plausibility_length 250
        uniqueness_length   1500
    }

}
SIMP

    # Patch meta_mode.info to remove its simplification overrides, which would
    # otherwise re-enable tip clipping, bulge removal, etc.
    python3 -c "
import re, pathlib
p = pathlib.Path('${CONFIGS_DIR}/debruijn/meta_mode.info')
text = p.read_text()
# Remove top-level simp { ... } and preliminary_simp { ... } blocks
def remove_block(t, name):
    pattern = r'^' + name + r'\s*\{'
    m = re.search(pattern, t, re.MULTILINE)
    if not m:
        return t
    start = m.start()
    depth = 0
    i = m.end() - 1
    while i < len(t):
        if t[i] == '{': depth += 1
        elif t[i] == '}': depth -= 1
        if depth == 0:
            return t[:start] + t[i+1:]
        i += 1
    return t
text = remove_block(text, 'simp')
text = remove_block(text, 'preliminary_simp')
p.write_text(text)
"
fi

# --- Run SPAdes with custom configs ---
docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint spades.py \
    quay.io/biocontainers/spades:4.2.0--h8d6e82b_2 \
    --meta \
    --only-assembler \
    --disable-rr \
    --configs-dir /output/custom_configs \
    --phred-offset 33 \
    -k "${K_LIST}" \
    -1 /input/sim_reads_R1.fastq \
    -2 /input/sim_reads_R2.fastq \
    -o /output \
    -t 4
