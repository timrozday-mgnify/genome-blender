//! PE and intra-read combo LMDB indexes.
//!
//! Schema (same for both PE and intra)
//! ------------------------------------
//! `"combo"` database (regular):
//!   key   = canonical combo hash (u64 LE, 8 bytes)
//!   value = occurrence count (u32 LE, 4 bytes)
//!
//! `"meta"` database:
//!   `n_reads`  → u64 LE
//!   `l`        → u64 LE
//!   `density`  → f64 LE (the combo density threshold)
//!
//! The canonical combo hash of two minimizer hashes `a` and `b` is computed
//! with a commutative splitmix64-derived mixing function.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::Path;

use anyhow::Result;
use lmdb::{Database, DatabaseFlags, Transaction, WriteFlags};
use rayon::prelude::*;

use super::lmdb::{open_db, write_meta, ShardWriter, MAP_SIZE};
use crate::minimizers::Minimizer;

// ── Combo hash ────────────────────────────────────────────────────────────────

/// Canonical commutative combo hash of two minimizer hashes.
///
/// Uses a splitmix64-derived mixing function; the result is the same
/// regardless of the order of `a` and `b`.
pub fn combo_hash(a: u64, b: u64) -> u64 {
    const CM1: u64 = 0x9e3779b97f4a7c15;
    const CM2: u64 = 0x6c62272e07bb0142;
    const CM3: u64 = 0x94d049bb133111eb;
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let h = lo ^ hi.wrapping_mul(CM1);
    let h = (h ^ (h >> 30)).wrapping_mul(CM2);
    let h = (h ^ (h >> 27)).wrapping_mul(CM3);
    h ^ (h >> 31)
}

// ── Intra-read combo ──────────────────────────────────────────────────────────

/// Compute all within-read minimizer pair hashes that pass `density`.
///
/// All C(n, 2) pairs of minimizers within a single read are considered.
pub fn intra_combo_hashes(minimizers: &[Minimizer], density: f64) -> Vec<u64> {
    let bound = (density * u64::MAX as f64) as u64;
    let n = minimizers.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let h = combo_hash(minimizers[i].hash, minimizers[j].hash);
            if h <= bound {
                out.push(h);
            }
        }
    }
    out
}

// ── PE combo spill file ───────────────────────────────────────────────────────

/// Write one R1 read's minimizer (hash, position) pairs to the spill stream.
///
/// Format: `[n: u32 LE] [(hash: u64 LE, pos: u32 LE) × n]`
pub fn spill_write(writer: &mut impl Write, minimizers: &[Minimizer]) -> io::Result<()> {
    let n = minimizers.len() as u32;
    writer.write_all(&n.to_le_bytes())?;
    for m in minimizers {
        writer.write_all(&m.hash.to_le_bytes())?;
        writer.write_all(&m.pos.to_le_bytes())?;
    }
    Ok(())
}

/// Read one R1 read's entry from the spill stream.
pub fn spill_read(reader: &mut impl Read) -> io::Result<Vec<Minimizer>> {
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        reader.read_exact(&mut buf4)?;
        out.push(Minimizer {
            hash: u64::from_le_bytes(buf8),
            pos: u32::from_le_bytes(buf4),
        });
    }
    Ok(out)
}

/// Compute PE combo hashes from paired R1 and R2 minimizers.
///
/// For each (R1 minimizer at `p1`, R2 minimizer at `p2`) where
/// `|p1 - p2| ≤ max_distance`, compute `combo_hash(h1, h2)`.
/// Retain only hashes ≤ `pe_density × u64::MAX`.
pub fn pe_combo_hashes(
    r1_minimizers: &[Minimizer],
    r2_minimizers: &[Minimizer],
    pe_density: f64,
    max_distance: u32,
) -> Vec<u64> {
    let bound = (pe_density * u64::MAX as f64) as u64;
    let mut out = Vec::new();
    for m1 in r1_minimizers {
        for m2 in r2_minimizers {
            let dist = m1.pos.abs_diff(m2.pos);
            if dist <= max_distance {
                let h = combo_hash(m1.hash, m2.hash);
                if h <= bound {
                    out.push(h);
                }
            }
        }
    }
    out
}

// ── Combo index writer ────────────────────────────────────────────────────────

/// Writer for a single sharded combo LMDB index (PE or intra).
pub struct ComboIndexWriter {
    shard: ShardWriter,
    combo_db: Database,
    meta_db: Database,
    /// In-memory count accumulator: combo_hash → count delta to flush.
    counts: HashMap<u64, u32>,
    batch_size: usize,
    n_reads_total: u64,
    l: u64,
    density: f64,
}

impl ComboIndexWriter {
    pub fn new(
        prefix: &Path,
        batch_size: usize,
        shard_size: u64,
        l: u64,
        density: f64,
    ) -> Result<Self> {
        let shard = ShardWriter::new(
            prefix,
            ".lmdb",
            "_shard_list",
            shard_size,
            MAP_SIZE,
            4,
        )?;
        let combo_db = open_db(&shard.env, "combo", DatabaseFlags::empty())?;
        let meta_db = open_db(&shard.env, "meta", DatabaseFlags::empty())?;
        Ok(ComboIndexWriter {
            shard,
            combo_db,
            meta_db,
            counts: HashMap::new(),
            batch_size,
            n_reads_total: 0,
            l,
            density,
        })
    }

    /// Accumulate combo hashes for one read pair / read.
    pub fn push_hashes(&mut self, hashes: &[u64]) -> Result<()> {
        self.n_reads_total += 1;
        for &h in hashes {
            *self.counts.entry(h).or_insert(0) += 1;
        }
        if self.counts.len() >= self.batch_size {
            self.flush_counts()?;
        }
        self.shard.tick(4)?;
        Ok(())
    }

    /// Flush accumulated counts to LMDB, merging with any existing values.
    fn flush_counts(&mut self) -> Result<()> {
        if self.counts.is_empty() {
            return Ok(());
        }
        // Sort hashes for APPEND optimisation.
        let mut sorted: Vec<([u8; 8], [u8; 4])> = self
            .counts
            .drain()
            .map(|(h, c)| (h.to_le_bytes(), c.to_le_bytes()))
            .collect();
        sorted.par_sort_unstable_by_key(|(k, _)| *k);

        // For keys that already exist in LMDB we must read-modify-write.
        // Read existing values for the overlap region (keys ≤ last existing key).
        let last = super::lmdb::last_key(&self.shard.env, self.combo_db)?;
        let (overlap, append): (&[_], &[_]) = match &last {
            None => (&[], &sorted),
            Some(lk) => {
                let split = sorted.partition_point(|(k, _)| k.as_ref() <= lk.as_slice());
                (&sorted[..split], &sorted[split..])
            }
        };

        let mut txn = self.shard.env.begin_rw_txn()?;
        for (k, v_delta) in overlap {
            let delta = u32::from_le_bytes(*v_delta);
            let existing: u32 = txn
                .get(self.combo_db, k)
                .ok()
                .and_then(|b| b.try_into().ok())
                .map(u32::from_le_bytes)
                .unwrap_or(0);
            let new_count = existing.saturating_add(delta);
            txn.put(self.combo_db, k, &new_count.to_le_bytes(), WriteFlags::empty())?;
        }
        for (k, v) in append {
            txn.put(self.combo_db, k, v, WriteFlags::APPEND)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Flush remaining counts and write metadata.
    pub fn finalize(mut self) -> Result<()> {
        self.flush_counts()?;
        let density_bits = self.density.to_bits().to_le_bytes();
        write_meta(
            &self.shard.env,
            self.meta_db,
            &[
                (b"n_reads".as_ref(), &self.n_reads_total.to_le_bytes()),
                (b"l".as_ref(), &self.l.to_le_bytes()),
                (b"density".as_ref(), &density_bits),
            ],
        )?;
        self.shard.finalize()
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimizers::Minimizer;
    use super::super::lmdb::{open_env, TEST_MAP_SIZE};
    use lmdb::{Transaction, DatabaseFlags};
    use tempfile::TempDir;

    fn m(hash: u64, pos: u32) -> Minimizer {
        Minimizer { hash, pos }
    }

    #[test]
    fn combo_hash_is_commutative() {
        let a = 0xdeadbeef12345678u64;
        let b = 0xfeedcafe87654321u64;
        assert_eq!(combo_hash(a, b), combo_hash(b, a));
    }

    #[test]
    fn combo_hash_differs_for_different_inputs() {
        assert_ne!(combo_hash(1, 2), combo_hash(1, 3));
        assert_ne!(combo_hash(1, 1), combo_hash(2, 2));
    }

    #[test]
    fn intra_combo_density_one_gives_all_pairs() {
        let mins = vec![m(1, 0), m(2, 5), m(3, 10)];
        // C(3,2) = 3 pairs; with density=1.0 all pass.
        let hashes = intra_combo_hashes(&mins, 1.0);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn intra_combo_density_zero_gives_none() {
        let mins = vec![m(1, 0), m(2, 5)];
        let hashes = intra_combo_hashes(&mins, 0.0);
        assert!(hashes.is_empty() || hashes.iter().all(|&h| h == 0));
    }

    #[test]
    fn pe_combo_distance_filter() {
        let r1 = vec![m(10, 5), m(20, 50)];
        let r2 = vec![m(30, 7)]; // pos 7 is close to r1[0].pos=5 (dist=2) but not r1[1].pos=50 (dist=43)
        let hashes = pe_combo_hashes(&r1, &r2, 1.0, 3);
        // Only pair (10, 30) passes distance ≤ 3.
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes[0], combo_hash(10, 30));
    }

    #[test]
    fn spill_round_trip() {
        let mins = vec![m(0xABCD, 10), m(0xEF01, 20)];
        let mut buf: Vec<u8> = Vec::new();
        spill_write(&mut buf, &mins).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let recovered = spill_read(&mut cursor).unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].hash, 0xABCD);
        assert_eq!(recovered[0].pos, 10);
        assert_eq!(recovered[1].hash, 0xEF01);
        assert_eq!(recovered[1].pos, 20);
    }

    #[test]
    fn combo_index_writer_accumulates_counts() {
        let dir = TempDir::new().unwrap();
        let prefix = dir.path().join("combo");
        let mut writer = ComboIndexWriter::new(&prefix, 100, 0, 17, 0.5).unwrap();
        // Push the same hash twice → count should be 2.
        let h = combo_hash(1, 2);
        writer.push_hashes(&[h]).unwrap();
        writer.push_hashes(&[h]).unwrap();
        writer.finalize().unwrap();

        let shard_path = dir.path().join("combo_shard_0.lmdb");
        let env = open_env(&shard_path, TEST_MAP_SIZE, 4).unwrap();
        let db = env.open_db(Some("combo")).unwrap();
        let txn = env.begin_ro_txn().unwrap();
        let val: &[u8] = txn.get(db, &h.to_le_bytes()).unwrap();
        let count = u32::from_le_bytes(val.try_into().unwrap());
        assert_eq!(count, 2);
    }
}
