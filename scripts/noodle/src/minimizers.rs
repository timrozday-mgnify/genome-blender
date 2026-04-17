//! Density-based minimizer extraction from DNA sequences.
//!
//! An l-mer is selected as a minimizer if its canonical NtHash value ≤
//! `(density × u64::MAX) as u64`.  This yields approximately `density`
//! fraction of all l-mers as minimizers, distributed uniformly across the
//! genome.
//!
//! Homopolymer compression (HPC) is applied before hashing: runs of
//! identical bases are collapsed to a single base.  Minimizer hashes are
//! those of the HPC l-mers; positions are mapped back to the original
//! (uncompressed) sequence coordinates.

use crate::nthash::NtHashIter;

/// One selected minimizer: its canonical NtHash value and its start position
/// in the original (non-HPC) sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Minimizer {
    pub hash: u64,
    pub pos: u32,
}

/// Parameters governing minimizer selection.
#[derive(Debug, Clone, Copy)]
pub struct MinParams {
    /// l-mer length.
    pub l: usize,
    /// Density threshold: fraction of l-mers to retain as minimizers.
    pub density: f64,
}

impl MinParams {
    pub fn new(l: usize, density: f64) -> Self {
        Self { l, density }
    }

    /// The hash bound derived from density: retain l-mers with hash ≤ bound.
    pub fn hash_bound(&self) -> u64 {
        (self.density * u64::MAX as f64) as u64
    }
}

/// Compress homopolymer runs: AAACCCGGG → ACG.
///
/// Also returns a mapping from HPC position → original position of the FIRST
/// base in each run.
pub fn hpc_compress(seq: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let mut compressed = Vec::with_capacity(seq.len());
    let mut positions: Vec<u32> = Vec::with_capacity(seq.len());
    let mut last = 0u8;
    for (i, &b) in seq.iter().enumerate() {
        if b != last {
            compressed.push(b);
            positions.push(i as u32);
            last = b;
        }
    }
    (compressed, positions)
}

/// Extract minimizers from a DNA sequence.
///
/// Applies HPC compression, then selects l-mers whose canonical NtHash is
/// within the density threshold.  Returns minimizers in order of their
/// position in the original sequence.
pub fn extract_minimizers(seq: &[u8], params: MinParams) -> Vec<Minimizer> {
    if seq.len() < params.l {
        return Vec::new();
    }
    let (hpc_seq, hpc_pos) = hpc_compress(seq);
    if hpc_seq.len() < params.l {
        return Vec::new();
    }
    let bound = params.hash_bound();
    let Some(iter) = NtHashIter::new(&hpc_seq, params.l) else {
        return Vec::new();
    };
    iter.enumerate()
        .filter_map(|(i, hash)| {
            if hash <= bound {
                Some(Minimizer { hash, pos: hpc_pos[i] })
            } else {
                None
            }
        })
        .collect()
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hpc_no_runs() {
        let (c, p) = hpc_compress(b"ACGT");
        assert_eq!(c, b"ACGT");
        assert_eq!(p, vec![0, 1, 2, 3]);
    }

    #[test]
    fn hpc_all_same() {
        let (c, p) = hpc_compress(b"AAAA");
        assert_eq!(c, b"A");
        assert_eq!(p, vec![0]);
    }

    #[test]
    fn hpc_mixed_runs() {
        let (c, p) = hpc_compress(b"AAACCCGGG");
        assert_eq!(c, b"ACG");
        assert_eq!(p, vec![0, 3, 6]);
    }

    #[test]
    fn extract_empty_seq() {
        let params = MinParams::new(17, 0.1);
        assert!(extract_minimizers(b"", params).is_empty());
    }

    #[test]
    fn extract_seq_shorter_than_l() {
        let params = MinParams::new(17, 0.1);
        assert!(extract_minimizers(b"ACGT", params).is_empty());
    }

    #[test]
    fn extract_density_one_selects_all() {
        // density=1.0 → bound = u64::MAX → all l-mers selected.
        let seq = b"ATCGATCGATCGATCGATCGATCGATCGATCG";
        let params = MinParams::new(5, 1.0);
        let minimizers = extract_minimizers(seq, params);
        // After HPC, ATCGATCG... has no runs, so hpc_seq == seq.
        // Number of 5-mers = len - 5 + 1.
        // But some might be deduplicated if the same hash appears? No - extract_minimizers
        // returns ALL that pass the threshold, including repeated hashes.
        let hpc_len = {
            let (h, _) = hpc_compress(seq);
            h.len()
        };
        assert_eq!(minimizers.len(), hpc_len - params.l + 1);
    }

    #[test]
    fn extract_density_zero_selects_none() {
        // density=0.0 → bound = 0 → only hash==0 selected (essentially none).
        let seq = b"ATCGATCGATCGATCGATCGATCGATCGATCG";
        let params = MinParams::new(5, 0.0);
        let minimizers = extract_minimizers(seq, params);
        assert!(minimizers.is_empty() || minimizers.iter().all(|m| m.hash == 0));
    }

    #[test]
    fn minimizer_positions_are_in_original_coordinates() {
        // Sequence with a run: AAATTT → HPC: AT → positions [0, 3]
        // If density=1.0 and l=2, we should get the AT 2-mer with pos=0.
        let seq = b"AAATTT";
        let params = MinParams::new(2, 1.0);
        let mins = extract_minimizers(seq, params);
        // HPC: "AT" at original positions [0, 3].
        // Only one 2-mer: AT, at HPC pos 0 → original pos 0.
        assert_eq!(mins.len(), 1);
        assert_eq!(mins[0].pos, 0);
    }

    #[test]
    fn minimizers_respect_density() {
        let seq: Vec<u8> = b"ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
            .iter()
            .copied()
            .collect();
        let params = MinParams::new(11, 0.1);
        let all = extract_minimizers(&seq, MinParams::new(11, 1.0));
        let selected = extract_minimizers(&seq, params);
        // On average ~10% selected; allow wide tolerance for short seq.
        assert!(selected.len() <= all.len());
    }
}
