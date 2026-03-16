/// genome_blender_core — fast per-base error assembly with rayon parallelism.
///
/// Exposed Python functions:
/// - ``apply_errors_batch``            — apply pre-computed errors (backward compat)
/// - ``sample_quality_scores_batch``   — Rust discrete-HMM quality sampler (opt 1)
/// - ``sample_and_apply_errors_batch`` — combined HMM + error assembly (opt 3)
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;

// ACGT as bytes
const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

// Substitution alternatives for each base index (A=0, C=1, G=2, T=3)
const ALTS: [[u8; 3]; 4] = [
    [b'C', b'G', b'T'], // alts for A
    [b'A', b'G', b'T'], // alts for C
    [b'A', b'C', b'T'], // alts for G
    [b'A', b'C', b'G'], // alts for T
];

/// Map an ASCII base byte to its index in ACGT (0-3).
/// Unknown bases map to 0 (A), the same fallback as Python.
#[inline(always)]
fn base_idx(b: u8) -> usize {
    match b.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 0,
    }
}

// ------------------------------------------------------------------
// Per-read seed derivation
// ------------------------------------------------------------------

/// splitmix64: mix base_seed with a read index to get a unique u64 seed.
#[inline(always)]
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e3779b97f4a7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[inline(always)]
fn derive_seed(base_seed: u64, idx: usize) -> u64 {
    splitmix64(base_seed.wrapping_add(idx as u64))
}

// ------------------------------------------------------------------
// Categorical sampling
// ------------------------------------------------------------------

/// Sample from categorical(softmax(logits)) using the stable CDF method.
/// Subtracts max logit for numerical stability before exponentiating.
#[inline]
fn sample_categorical(logits: &[f32], rng: &mut SmallRng) -> usize {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;
    let exps: Vec<f32> = logits
        .iter()
        .map(|&l| {
            let e = (l - max_l).exp();
            total += e;
            e
        })
        .collect();
    let threshold = rng.gen::<f32>() * total;
    let mut cumsum = 0.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        cumsum += e;
        if cumsum >= threshold {
            return i;
        }
    }
    logits.len() - 1
}

// ------------------------------------------------------------------
// Discrete HMM quality sampler
// ------------------------------------------------------------------

/// Sample a quality-score trace of *length* positions from a discrete HMM.
///
/// # Parameters
/// * `initial_logits`    — flat `(num_states,)` f32 slice
/// * `transition_logits` — flat `(num_states × num_states,)` row-major f32 slice
/// * `emission_logits`   — flat `(num_states × num_quality_values,)` row-major f32 slice
fn sample_quality_hmm(
    initial_logits: &[f32],
    transition_logits: &[f32],
    emission_logits: &[f32],
    num_states: usize,
    num_quality_values: usize,
    length: usize,
    rng: &mut SmallRng,
) -> Vec<u8> {
    let mut qualities = Vec::with_capacity(length);
    let mut state = sample_categorical(initial_logits, rng);

    for _ in 0..length {
        // Emit quality from current state
        let em_start = state * num_quality_values;
        let q = sample_categorical(
            &emission_logits[em_start..em_start + num_quality_values],
            rng,
        );
        qualities.push(q as u8);

        // Transition to next state
        let tr_start = state * num_states;
        state = sample_categorical(
            &transition_logits[tr_start..tr_start + num_states],
            rng,
        );
    }
    qualities
}

// ------------------------------------------------------------------
// Quality calibration
// ------------------------------------------------------------------

enum CalibrationKind {
    Phred,
    LogLinear {
        intercept: f64,
        slope: f64,
        floor: f64,
        ceiling: f64,
    },
    Sigmoid {
        steepness: f64,
        midpoint: f64,
        floor: f64,
        ceiling: f64,
    },
}

/// Convert a Phred quality score to an error probability, scaled and clamped.
#[inline(always)]
fn p_error_from_q(cal: &CalibrationKind, q: u8, scale: f32) -> f64 {
    let q = q as f64;
    let p = match cal {
        CalibrationKind::Phred => 10.0_f64.powf(-q / 10.0),
        CalibrationKind::LogLinear {
            intercept,
            slope,
            floor,
            ceiling,
        } => 10.0_f64.powf(intercept + slope * q).clamp(*floor, *ceiling),
        CalibrationKind::Sigmoid {
            steepness,
            midpoint,
            floor,
            ceiling,
        } => {
            let x = -steepness * (q - midpoint);
            let sig = 1.0 / (1.0 + (-x).exp());
            floor + (ceiling - floor) * sig
        }
    };
    (p * scale as f64).clamp(0.0, 1.0)
}

/// Parse calibration type string + params slice into a ``CalibrationKind``.
///
/// * ``"log-linear"`` — params: `[intercept, slope, floor, ceiling]`
/// * ``"sigmoid"``    — params: `[steepness, midpoint, floor, ceiling]`
/// * anything else   → Phred (theoretical)
fn parse_calibration(cal_type: &str, params: &[f64]) -> CalibrationKind {
    match cal_type {
        "log-linear" => CalibrationKind::LogLinear {
            intercept: *params.first().unwrap_or(&-0.3),
            slope:     *params.get(1).unwrap_or(&-0.08),
            floor:     *params.get(2).unwrap_or(&1e-7),
            ceiling:   *params.get(3).unwrap_or(&0.5),
        },
        "sigmoid" => CalibrationKind::Sigmoid {
            steepness: *params.first().unwrap_or(&0.25),
            midpoint:  *params.get(1).unwrap_or(&15.0),
            floor:     *params.get(2).unwrap_or(&1e-6),
            ceiling:   *params.get(3).unwrap_or(&0.5),
        },
        _ => CalibrationKind::Phred,
    }
}

// ------------------------------------------------------------------
// CIGAR run-length encoding
// ------------------------------------------------------------------

fn rle_cigar(cigar_ops: &[u8]) -> Vec<(u8, u32)> {
    let mut tuples: Vec<(u8, u32)> = Vec::new();
    if !cigar_ops.is_empty() {
        let mut cur = cigar_ops[0];
        let mut run: u32 = 1;
        for &op in &cigar_ops[1..] {
            if op == cur {
                run += 1;
            } else {
                tuples.push((cur, run));
                cur = op;
                run = 1;
            }
        }
        tuples.push((cur, run));
    }
    tuples
}

// ------------------------------------------------------------------
// Assembly from pre-computed decision arrays (apply_errors_batch path)
// ------------------------------------------------------------------

fn assemble_read(
    seq: &[u8],
    q_ints: &[i64],
    is_error: &[bool],
    error_types: &[i64],
    sub_choices: &[i64],
    ins_bases: &[i64],
) -> (String, String, Vec<(u8, u32)>) {
    let ref_len = seq.len();
    let cap = ref_len + 8;
    let mut modified: Vec<u8> = Vec::with_capacity(cap);
    let mut qual_chars: Vec<u8> = Vec::with_capacity(cap);
    let mut cigar_ops: Vec<u8> = Vec::with_capacity(cap);

    for pos in 0..ref_len {
        let q_char = (q_ints[pos] as u8).wrapping_add(33);

        if !is_error[pos] {
            modified.push(seq[pos]);
            qual_chars.push(q_char);
            cigar_ops.push(0); // M
        } else {
            match error_types[pos] {
                0 => {
                    let orig = base_idx(seq[pos]);
                    let alt = ALTS[orig][sub_choices[pos] as usize];
                    modified.push(alt);
                    qual_chars.push(q_char);
                    cigar_ops.push(0); // M
                }
                1 => {
                    modified.push(BASES[ins_bases[pos] as usize]);
                    qual_chars.push(q_char);
                    cigar_ops.push(1); // I
                    modified.push(seq[pos]);
                    qual_chars.push(q_char);
                    cigar_ops.push(0); // M
                }
                _ => {
                    cigar_ops.push(2); // D
                }
            }
        }
    }

    let cigar_tuples = rle_cigar(&cigar_ops);
    // Safety: modified/qual_chars contain only ASCII bytes we wrote
    let new_seq = unsafe { String::from_utf8_unchecked(modified) };
    let qual_str = unsafe { String::from_utf8_unchecked(qual_chars) };
    (new_seq, qual_str, cigar_tuples)
}

// ------------------------------------------------------------------
// Combined HMM sampling + error assembly per read
// ------------------------------------------------------------------

/// Sample quality scores from the HMM then apply position-wise errors,
/// all in a single pass with its own ``SmallRng``.
fn process_read(
    seq: &[u8],
    initial_logits: &[f32],
    transition_logits: &[f32],
    emission_logits: &[f32],
    num_states: usize,
    num_quality_values: usize,
    sub_ratio: f32,
    ins_ratio: f32,
    error_rate_scale: f32,
    calibration: &CalibrationKind,
    rng: &mut SmallRng,
) -> (String, String, Vec<(u8, u32)>) {
    let ref_len = seq.len();

    // Step 1: sample quality scores via HMM
    let qualities = sample_quality_hmm(
        initial_logits,
        transition_logits,
        emission_logits,
        num_states,
        num_quality_values,
        ref_len,
        rng,
    );

    // Step 2: apply position-wise errors
    let cap = ref_len + 8;
    let mut modified: Vec<u8> = Vec::with_capacity(cap);
    let mut qual_chars: Vec<u8> = Vec::with_capacity(cap);
    let mut cigar_ops: Vec<u8> = Vec::with_capacity(cap);

    // Cumulative error-type thresholds (deletion absorbs the remainder)
    let thresh_sub = sub_ratio;
    let thresh_ins = sub_ratio + ins_ratio;

    for pos in 0..ref_len {
        let q = qualities[pos];
        let q_char = q.wrapping_add(33);
        let p_err = p_error_from_q(calibration, q, error_rate_scale);

        if rng.gen::<f64>() >= p_err {
            // No error
            modified.push(seq[pos]);
            qual_chars.push(q_char);
            cigar_ops.push(0); // M
        } else {
            let r: f32 = rng.gen();
            if r < thresh_sub {
                // Substitution: replace with a different base
                let orig = base_idx(seq[pos]);
                let alt_idx = (rng.gen::<u32>() % 3) as usize;
                modified.push(ALTS[orig][alt_idx]);
                qual_chars.push(q_char);
                cigar_ops.push(0); // M
            } else if r < thresh_ins {
                // Insertion: random base before the reference base
                let ins_base = BASES[(rng.gen::<u32>() % 4) as usize];
                modified.push(ins_base);
                qual_chars.push(q_char);
                cigar_ops.push(1); // I
                modified.push(seq[pos]);
                qual_chars.push(q_char);
                cigar_ops.push(0); // M
            } else {
                // Deletion: skip reference base
                cigar_ops.push(2); // D
            }
        }
    }

    let cigar_tuples = rle_cigar(&cigar_ops);
    let new_seq = unsafe { String::from_utf8_unchecked(modified) };
    let qual_str = unsafe { String::from_utf8_unchecked(qual_chars) };
    (new_seq, qual_str, cigar_tuples)
}

// ------------------------------------------------------------------
// PyO3 functions
// ------------------------------------------------------------------

/// Apply pre-computed sequencing errors to a batch of reads in parallel.
///
/// All stochastic decision arrays are flat (concatenated across reads)
/// and sliced using the per-read ``lengths``.
///
/// # Parameters (Python-visible)
/// * ``sequences``   — list of reference sequence strings
/// * ``q_ints``      — flat list of int quality scores (sum of lengths)
/// * ``is_error``    — flat list of bool (sum of lengths)
/// * ``error_types`` — flat list of int 0/1/2 (sum of lengths)
/// * ``sub_choices`` — flat list of int 0/1/2 (sum of lengths)
/// * ``ins_bases``   — flat list of int 0–3 (sum of lengths)
/// * ``lengths``     — per-read lengths (len == number of reads)
///
/// # Returns
/// List of ``(new_seq, qual_str, cigar)`` tuples where ``cigar`` is a
/// list of ``(op, length)`` pairs (op: 0=M, 1=I, 2=D).
#[pyfunction]
fn apply_errors_batch(
    py: Python<'_>,
    sequences: Vec<String>,
    q_ints: Vec<i64>,
    is_error: Vec<bool>,
    error_types: Vec<i64>,
    sub_choices: Vec<i64>,
    ins_bases: Vec<i64>,
    lengths: Vec<usize>,
) -> PyResult<Vec<(String, String, Vec<(u8, u32)>)>> {
    let mut offsets: Vec<usize> = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0);
    for &l in &lengths {
        offsets.push(offsets.last().unwrap() + l);
    }

    let n = sequences.len();
    let result: Vec<(String, String, Vec<(u8, u32)>)> =
        py.allow_threads(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let start = offsets[i];
                    let end = offsets[i + 1];
                    assemble_read(
                        sequences[i].as_bytes(),
                        &q_ints[start..end],
                        &is_error[start..end],
                        &error_types[start..end],
                        &sub_choices[start..end],
                        &ins_bases[start..end],
                    )
                })
                .collect()
        });
    Ok(result)
}

/// Sample quality scores for a batch of reads from a discrete HMM in parallel.
///
/// Each read gets an independent ``SmallRng`` seeded from
/// ``splitmix64(base_seed + read_index)``.
///
/// # Parameters (Python-visible)
/// * ``initial_logits``    — flat f32 list of length ``num_states``
/// * ``transition_logits`` — flat f32 list of length ``num_states²`` (row-major)
/// * ``emission_logits``   — flat f32 list of length ``num_states × num_quality_values`` (row-major)
/// * ``num_states``        — number of HMM states
/// * ``num_quality_values``— number of distinct quality values (typically 94)
/// * ``lengths``           — per-read lengths
/// * ``base_seed``         — base seed for per-read RNG derivation
///
/// # Returns
/// List of lists of integer quality scores (Phred 0–93), one per read.
#[pyfunction]
fn sample_quality_scores_batch(
    py: Python<'_>,
    initial_logits: Vec<f32>,
    transition_logits: Vec<f32>,
    emission_logits: Vec<f32>,
    num_states: usize,
    num_quality_values: usize,
    lengths: Vec<usize>,
    base_seed: u64,
) -> PyResult<Vec<Vec<i64>>> {
    let result: Vec<Vec<i64>> = py.allow_threads(|| {
        (0..lengths.len())
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(derive_seed(base_seed, i));
                let qs = sample_quality_hmm(
                    &initial_logits,
                    &transition_logits,
                    &emission_logits,
                    num_states,
                    num_quality_values,
                    lengths[i],
                    &mut rng,
                );
                qs.iter().map(|&q| q as i64).collect()
            })
            .collect()
    });
    Ok(result)
}

/// Sample quality scores and apply errors to a batch of reads in one parallel pass.
///
/// Combines HMM quality sampling and error application in a single rayon-parallel
/// call, eliminating all Python intermediary data structures.
///
/// # Parameters (Python-visible)
/// * ``sequences``          — list of reference sequence strings
/// * ``initial_logits``     — flat f32 list of length ``num_states``
/// * ``transition_logits``  — flat f32 list of length ``num_states²`` (row-major)
/// * ``emission_logits``    — flat f32 list of length ``num_states × num_quality_values`` (row-major)
/// * ``num_states``         — number of HMM states
/// * ``num_quality_values`` — number of distinct quality values (typically 94)
/// * ``sub_ratio``          — substitution fraction of errors
/// * ``ins_ratio``          — insertion fraction of errors
/// * ``error_rate_scale``   — multiplicative scale on error probability
/// * ``calibration_type``   — ``"phred"``, ``"log-linear"``, or ``"sigmoid"``
/// * ``calibration_params`` — model-specific parameters (see ``parse_calibration``)
/// * ``base_seed``          — base seed for per-read RNG derivation
///
/// # Returns
/// List of ``(new_seq, qual_str, cigar)`` tuples.
#[pyfunction]
fn sample_and_apply_errors_batch(
    py: Python<'_>,
    sequences: Vec<String>,
    initial_logits: Vec<f32>,
    transition_logits: Vec<f32>,
    emission_logits: Vec<f32>,
    num_states: usize,
    num_quality_values: usize,
    sub_ratio: f32,
    ins_ratio: f32,
    error_rate_scale: f32,
    calibration_type: String,
    calibration_params: Vec<f64>,
    base_seed: u64,
) -> PyResult<Vec<(String, String, Vec<(u8, u32)>)>> {
    let result: Vec<(String, String, Vec<(u8, u32)>)> = py.allow_threads(|| {
        let cal = parse_calibration(&calibration_type, &calibration_params);
        sequences
            .par_iter()
            .enumerate()
            .map(|(i, seq)| {
                let mut rng = SmallRng::seed_from_u64(derive_seed(base_seed, i));
                process_read(
                    seq.as_bytes(),
                    &initial_logits,
                    &transition_logits,
                    &emission_logits,
                    num_states,
                    num_quality_values,
                    sub_ratio,
                    ins_ratio,
                    error_rate_scale,
                    &cal,
                    &mut rng,
                )
            })
            .collect()
    });
    Ok(result)
}

#[pymodule]
fn genome_blender_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_errors_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sample_quality_scores_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sample_and_apply_errors_batch, m)?)?;
    Ok(())
}
