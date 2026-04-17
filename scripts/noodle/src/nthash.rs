//! Canonical NtHash rolling hash for DNA l-mers.
//!
//! Implements the algorithm from Mohamadi et al. (2016) "ntHash: recursive
//! nucleotide hashing". The canonical hash at each position is the minimum of
//! the forward and reverse-complement hashes, ensuring strand-agnostic
//! minimizer selection.

/// NtHash seed values for A, C, G, T (from Mohamadi et al. 2016).
const SEED_A: u64 = 0x3c8bfbb395c60474;
const SEED_C: u64 = 0x3193c18562a02b4c;
const SEED_G: u64 = 0x20323ed082572324;
const SEED_T: u64 = 0x295549f54be24456;

/// Hash seed for a nucleotide (both cases). Returns 0 for N / ambiguous bases.
const fn seed(b: u8) -> u64 {
    match b {
        b'A' | b'a' => SEED_A,
        b'C' | b'c' => SEED_C,
        b'G' | b'g' => SEED_G,
        b'T' | b't' => SEED_T,
        _ => 0,
    }
}

/// Hash seed for the complement of a nucleotide.
const fn seed_rc(b: u8) -> u64 {
    match b {
        b'A' | b'a' => SEED_T,
        b'C' | b'c' => SEED_G,
        b'G' | b'g' => SEED_C,
        b'T' | b't' => SEED_A,
        _ => 0,
    }
}

const fn is_valid_base(b: u8) -> bool {
    matches!(b, b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't')
}

/// Iterator over canonical NtHash values for consecutive l-mers of `seq`.
///
/// Windows containing ambiguous bases (N or other non-ACGT) are silently
/// skipped — the rolling hash remains valid for subsequent windows.
#[derive(Debug)]
pub struct NtHashIter<'a> {
    seq: &'a [u8],
    l: usize,
    fwd: u64,
    rev: u64,
    n_in_window: usize,
    pos: usize,
}

impl<'a> NtHashIter<'a> {
    /// Create an iterator over l-mers of `seq`.
    ///
    /// Returns `None` if `seq` is shorter than `l` or `l` is zero.
    pub fn new(seq: &'a [u8], l: usize) -> Option<Self> {
        if l == 0 || seq.len() < l {
            return None;
        }
        let mut fwd = 0u64;
        let mut rev = 0u64;
        let mut n_in_window = 0usize;
        for i in 0..l {
            let b = seq[i];
            fwd ^= seed(b).rotate_left((l - 1 - i) as u32);
            rev ^= seed_rc(b).rotate_left(i as u32);
            if !is_valid_base(b) {
                n_in_window += 1;
            }
        }
        Some(NtHashIter { seq, l, fwd, rev, n_in_window, pos: 0 })
    }
}

impl<'a> Iterator for NtHashIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        loop {
            if self.pos + self.l > self.seq.len() {
                return None;
            }

            let canonical = self.fwd.min(self.rev);
            let valid = self.n_in_window == 0;

            // Advance the rolling hash to the next position if possible.
            if self.pos + self.l < self.seq.len() {
                let out = self.seq[self.pos];
                let inc = self.seq[self.pos + self.l];

                // Forward: H_f(i+1) = rol(H_f(i), 1) ^ rol(h(out), l) ^ h(inc)
                self.fwd = self.fwd.rotate_left(1)
                    ^ seed(out).rotate_left(self.l as u32)
                    ^ seed(inc);

                // RC: H_rc(i+1) = ror(H_rc(i), 1) ^ ror(h_rc(out), 1) ^ rol(h_rc(inc), l-1)
                self.rev = self.rev.rotate_right(1)
                    ^ seed_rc(out).rotate_right(1)
                    ^ seed_rc(inc).rotate_left((self.l - 1) as u32);

                // Update N count.
                if !is_valid_base(out) {
                    self.n_in_window -= 1;
                }
                if !is_valid_base(inc) {
                    self.n_in_window += 1;
                }
            }

            self.pos += 1;

            if valid {
                return Some(canonical);
            }
            // If the window contained an N, skip and try the next position.
        }
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_seq_returns_none() {
        assert!(NtHashIter::new(b"", 3).is_none());
    }

    #[test]
    fn seq_shorter_than_l_returns_none() {
        assert!(NtHashIter::new(b"ACG", 5).is_none());
    }

    #[test]
    fn single_lmer_yields_one_hash() {
        let hashes: Vec<u64> = NtHashIter::new(b"ACGT", 4).unwrap().collect();
        assert_eq!(hashes.len(), 1);
    }

    #[test]
    fn n_lmers_yield_n_plus_one_minus_l_hashes() {
        let seq = b"ATCGATCGATCG";
        let l = 4;
        let expected = seq.len() - l + 1;
        let hashes: Vec<u64> = NtHashIter::new(seq, l).unwrap().collect();
        assert_eq!(hashes.len(), expected);
    }

    #[test]
    fn canonical_is_symmetric() {
        // For a palindrome, forward == rc, so both orderings give the same hash.
        // For any sequence, hash(seq) == hash(rc(seq)).
        let seq  = b"ACGTTTTACGT";
        let rc   = b"ACGTAAAACGT";
        let h1: Vec<u64> = NtHashIter::new(seq, 11).unwrap().collect();
        let h2: Vec<u64> = NtHashIter::new(rc, 11).unwrap().collect();
        // Canonical of seq == canonical of its RC (single l-mer covering both).
        assert_eq!(h1, h2);
    }

    #[test]
    fn n_base_windows_are_skipped() {
        // "ACNGT" with l=3: windows are ACN, CNG, NGT.
        // All contain N → 0 valid hashes.
        let hashes: Vec<u64> = NtHashIter::new(b"ACNGT", 3).unwrap().collect();
        assert_eq!(hashes.len(), 0);
    }

    #[test]
    fn hash_zero_l_returns_none() {
        assert!(NtHashIter::new(b"ACGT", 0).is_none());
    }

    #[test]
    fn rolling_hash_matches_naive() {
        // Compare rolling hash to independently computed per-window hash.
        fn naive_hash(seq: &[u8], l: usize) -> Vec<u64> {
            (0..=seq.len() - l)
                .filter(|&i| seq[i..i + l].iter().all(|&b| is_valid_base(b)))
                .map(|i| {
                    let mut fwd = 0u64;
                    let mut rev = 0u64;
                    for j in 0..l {
                        fwd ^= seed(seq[i + j]).rotate_left((l - 1 - j) as u32);
                        rev ^= seed_rc(seq[i + j]).rotate_left(j as u32);
                    }
                    fwd.min(rev)
                })
                .collect()
        }

        let seq = b"ATCGATCGATCGATCG";
        let l = 5;
        let rolling: Vec<u64> = NtHashIter::new(seq, l).unwrap().collect();
        let naive = naive_hash(seq, l);
        assert_eq!(rolling, naive);
    }
}
