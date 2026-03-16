/// genome_blender_core — fast per-base error assembly with rayon parallelism.
///
/// Exposed Python function: ``apply_errors_batch``
///
/// All stochastic decisions (which positions are errors, which error
/// type, which substitute base, which insert base) are pre-computed by
/// the Python caller using vectorised torch operations and passed in as
/// flat arrays.  This module only performs the sequential per-read
/// assembly (the hot loop that was previously Python) and runs
/// independent reads in parallel via rayon.
use pyo3::prelude::*;
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
/// Unknown bases map to 0 (A), which is the same fallback as Python.
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

/// Assemble the modified sequence, quality string, and run-length
/// encoded CIGAR for a single read, given pre-computed decision arrays.
///
/// # Parameters
/// * `seq`        — reference sequence bytes for this read
/// * `q_ints`     — quality scores (Phred values, 0–93)
/// * `is_error`   — whether each reference position has an error
/// * `error_types`— 0=substitution, 1=insertion, 2=deletion
/// * `sub_choices`— 0/1/2 → which of the three substitution alts
/// * `ins_bases`  — 0–3 → which of ACGT is inserted
fn assemble_read(
    seq: &[u8],
    q_ints: &[i64],
    is_error: &[bool],
    error_types: &[i64],
    sub_choices: &[i64],
    ins_bases: &[i64],
) -> (String, String, Vec<(u8, u32)>) {
    let ref_len = seq.len();
    // Upper bound: insertions add one extra byte per position
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
                    // Substitution: replace with an alternative base
                    let orig = base_idx(seq[pos]);
                    let alt = ALTS[orig][sub_choices[pos] as usize];
                    modified.push(alt);
                    qual_chars.push(q_char);
                    cigar_ops.push(0); // M
                }
                1 => {
                    // Insertion: emit inserted base then reference base
                    modified.push(BASES[ins_bases[pos] as usize]);
                    qual_chars.push(q_char);
                    cigar_ops.push(1); // I
                    modified.push(seq[pos]);
                    qual_chars.push(q_char);
                    cigar_ops.push(0); // M
                }
                _ => {
                    // Deletion: skip reference base, record D op
                    cigar_ops.push(2); // D
                }
            }
        }
    }

    // Run-length encode CIGAR operations
    let mut cigar_tuples: Vec<(u8, u32)> = Vec::new();
    if !cigar_ops.is_empty() {
        let mut current_op = cigar_ops[0];
        let mut run: u32 = 1;
        for &op in &cigar_ops[1..] {
            if op == current_op {
                run += 1;
            } else {
                cigar_tuples.push((current_op, run));
                current_op = op;
                run = 1;
            }
        }
        cigar_tuples.push((current_op, run));
    }

    // Safety: modified contains only ASCII printable bytes we wrote
    let new_seq = unsafe { String::from_utf8_unchecked(modified) };
    let qual_str = unsafe { String::from_utf8_unchecked(qual_chars) };

    (new_seq, qual_str, cigar_tuples)
}

/// Apply pre-computed sequencing errors to a batch of reads in parallel.
///
/// All stochastic decision arrays are flat (concatenated across reads)
/// and sliced using the per-read ``lengths``.
///
/// # Parameters (Python-visible)
/// * ``sequences``      — list of reference sequence strings
/// * ``q_ints``         — flat list of int quality scores (sum of lengths)
/// * ``is_error``       — flat list of bool (sum of lengths)
/// * ``error_types``    — flat list of int 0/1/2 (sum of lengths)
/// * ``sub_choices``    — flat list of int 0/1/2 (sum of lengths)
/// * ``ins_bases``      — flat list of int 0–3 (sum of lengths)
/// * ``lengths``        — per-read lengths (len == number of reads)
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
    // Build per-read slices (offsets into flat arrays)
    let mut offsets: Vec<usize> = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0);
    for &l in &lengths {
        offsets.push(offsets.last().unwrap() + l);
    }

    let n = sequences.len();

    // Release the GIL while doing the heavy parallel work
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

#[pymodule]
fn genome_blender_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_errors_batch, m)?)?;
    Ok(())
}
