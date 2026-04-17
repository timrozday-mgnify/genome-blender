//! Counting bloom filter using FNV-1a 64-bit hashing with double-hashing to
//! derive 3 bit positions.
//!
//! On-disk format (compatible with bin/rust-mdbg):
//!
//! ```text
//! Offset  Bytes  Field
//! 0       8      Magic (type-specific, e.g. b"MNBLOOMB")
//! 8       8      Version (u64 LE = 1)
//! 16      8      Extra field (e.g. l-mer length, or density as f64) u64 LE
//! 24      8      n_bits (u64 LE)
//! 32      8      n_hash_fns (u64 LE = 3)
//! 40      n_bits/8  packed bit array (u64 LE words)
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// Default bloom filter size: 2^30 bits ≈ 128 MB, ~1% FPR for up to ~100 M
/// distinct elements with 3 hash functions.
pub const DEFAULT_BLOOM_BITS: usize = 1 << 30;

/// Number of independent hash positions per element.
const N_HASH_FNS: u64 = 3;

/// 64-bit FNV-1a hash of raw bytes.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET_BASIS;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// A simple Bloom filter backed by a `Vec<u64>` bit array.
#[derive(Debug)]
pub struct BloomFilter {
    bits: Vec<u64>,
    n_bits: usize,
}

impl BloomFilter {
    /// Create a new, empty bloom filter with the given number of bits.
    ///
    /// `n_bits` is rounded up to the next multiple of 64.
    pub fn new(n_bits: usize) -> Self {
        let n_words = (n_bits + 63) / 64;
        BloomFilter { bits: vec![0u64; n_words], n_bits: n_words * 64 }
    }

    /// Insert a u64 value.
    pub fn insert_u64(&mut self, value: u64) {
        let data = value.to_le_bytes();
        self.insert_raw(&data);
    }

    /// Insert arbitrary bytes.
    pub fn insert_raw(&mut self, data: &[u8]) {
        let (h1, h2) = double_hash(data);
        for i in 0..N_HASH_FNS {
            let pos = h1.wrapping_add(i.wrapping_mul(h2)) as usize % self.n_bits;
            self.bits[pos >> 6] |= 1u64 << (pos & 63);
        }
    }

    /// Test whether a u64 value is present (may false-positive).
    pub fn contains_u64(&self, value: u64) -> bool {
        let data = value.to_le_bytes();
        self.contains_raw(&data)
    }

    /// Test whether arbitrary bytes are present.
    pub fn contains_raw(&self, data: &[u8]) -> bool {
        let (h1, h2) = double_hash(data);
        (0..N_HASH_FNS).all(|i| {
            let pos = h1.wrapping_add(i.wrapping_mul(h2)) as usize % self.n_bits;
            self.bits[pos >> 6] & (1u64 << (pos & 63)) != 0
        })
    }

    /// Serialize to disk with the given `magic` bytes and `extra` field.
    ///
    /// `extra` is written as the third u64 in the header (used for l-mer
    /// length or density depending on the filter type).
    pub fn write_to_file(&self, path: &Path, magic: &[u8; 8], extra: u64) -> Result<()> {
        let mut f = BufWriter::new(
            File::create(path).with_context(|| format!("creating bloom filter {:?}", path))?,
        );
        f.write_all(magic)?;
        f.write_all(&1u64.to_le_bytes())?;         // version
        f.write_all(&extra.to_le_bytes())?;         // extra (l / density bits)
        f.write_all(&(self.n_bits as u64).to_le_bytes())?;
        f.write_all(&N_HASH_FNS.to_le_bytes())?;
        for word in &self.bits {
            f.write_all(&word.to_le_bytes())?;
        }
        f.flush()?;
        Ok(())
    }
}

/// Double-hashing: derive two independent 64-bit hashes from data.
fn double_hash(data: &[u8]) -> (u64, u64) {
    let h1 = fnv1a_64(data);
    let h2 = h1.rotate_left(17).wrapping_mul(0x9e3779b97f4a7c15u64);
    (h1, h2)
}

// ── Magic byte constants ──────────────────────────────────────────────────────

pub const MAGIC_MINIMIZER_BLOOM: &[u8; 8] = b"MNBLOOMB";
pub const MAGIC_PE_COMBO_BLOOM: &[u8; 8] = b"PECMBBFL";
pub const MAGIC_INTRA_COMBO_BLOOM: &[u8; 8] = b"ICMBBFL1";

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn inserted_items_are_present() {
        let mut bf = BloomFilter::new(1 << 20);
        let hashes: Vec<u64> = (0..1000).map(|i| i * 7919 + 13).collect();
        for &h in &hashes {
            bf.insert_u64(h);
        }
        for &h in &hashes {
            assert!(bf.contains_u64(h), "false negative for {h}");
        }
    }

    #[test]
    fn false_positive_rate_is_low() {
        let mut bf = BloomFilter::new(1 << 20);
        for i in 0u64..1000 {
            bf.insert_u64(i);
        }
        let fp: usize = (10_000u64..20_000u64)
            .filter(|&x| bf.contains_u64(x))
            .count();
        // Expect < 5% FPR with 1M bits and 1000 elements.
        assert!(fp < 500, "FPR too high: {fp} / 10000");
    }

    #[test]
    fn serialization_round_trip() {
        let mut bf = BloomFilter::new(1 << 16);
        for i in 0u64..100 {
            bf.insert_u64(i * 12345);
        }
        let tmp = NamedTempFile::new().unwrap();
        bf.write_to_file(tmp.path(), MAGIC_MINIMIZER_BLOOM, 17).unwrap();

        // Verify header magic is correct.
        let data = std::fs::read(tmp.path()).unwrap();
        assert_eq!(&data[..8], b"MNBLOOMB");
        // Version = 1
        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()), 1);
        // Extra = 17
        assert_eq!(u64::from_le_bytes(data[16..24].try_into().unwrap()), 17);
        // n_bits
        let n_bits = u64::from_le_bytes(data[24..32].try_into().unwrap()) as usize;
        assert_eq!(n_bits, bf.n_bits);
        // n_hash_fns = 3
        assert_eq!(u64::from_le_bytes(data[32..40].try_into().unwrap()), 3);
        // Bit array size
        assert_eq!(data.len(), 40 + n_bits / 8);
    }
}
