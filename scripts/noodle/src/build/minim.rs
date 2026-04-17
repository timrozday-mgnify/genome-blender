//! Minimizer LMDB index: inverted map from minimizer hash → read IDs.
//!
//! Schema
//! ------
//! `"minimizers"` database (DUPSORT + DUPFIXED):
//!   key   = minimizer hash (u64 LE, 8 bytes)
//!   value = read ID (u32 LE, 4 bytes)  — one entry per (hash, read_id) pair
//!
//! `"meta"` database:
//!   `n_reads`       → u64 LE
//!   `l`             → u32 LE
//!   `read_id_width` → u32 LE = 4
//!
//! Only minimizers whose occurrence count across all reads is ≥ `min_abund`
//! are written to the index.

use std::collections::HashMap;

use anyhow::Result;
use lmdb::{Database, DatabaseFlags, WriteFlags};
use rayon::prelude::*;

use super::lmdb::{flush_sorted_batch, open_db, write_meta, ShardWriter, MAP_SIZE};
use crate::minimizers::Minimizer;

/// Encodes a (minimizer_hash, read_id) pair as sortable key and value bytes.
#[inline]
fn entry_bytes(hash: u64, read_id: u32) -> ([u8; 8], [u8; 4]) {
    (hash.to_le_bytes(), read_id.to_le_bytes())
}

type MinimBatch = Vec<([u8; 8], [u8; 4])>;

/// Writer for the sharded minimizer LMDB index.
pub struct MinimIndexWriter {
    shard: ShardWriter,
    min_db: Database,
    meta_db: Database,
    batch: MinimBatch,
    batch_size: usize,
    n_reads_total: u64,
    l: u32,
    counts: HashMap<u64, u32>,
    min_abund: u32,
}

impl MinimIndexWriter {
    pub fn new(
        prefix: &std::path::Path,
        batch_size: usize,
        shard_size: u64,
        l: u32,
        min_abund: u32,
    ) -> Result<Self> {
        let shard = ShardWriter::new(
            prefix,
            ".lmdb",
            "_shard_list",
            shard_size,
            MAP_SIZE,
            4,
        )?;
        let min_db = open_db(
            &shard.env,
            "minimizers",
            DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED,
        )?;
        let meta_db = open_db(&shard.env, "meta", DatabaseFlags::empty())?;
        Ok(MinimIndexWriter {
            shard,
            min_db,
            meta_db,
            batch: Vec::with_capacity(batch_size),
            batch_size,
            n_reads_total: 0,
            l,
            counts: HashMap::new(),
            min_abund,
        })
    }

    /// Record the minimizers of a read for occurrence counting (first pass).
    ///
    /// Call this once per read before the second pass with [`push`].
    pub fn count(&mut self, minimizers: &[Minimizer]) {
        for m in minimizers {
            *self.counts.entry(m.hash).or_insert(0) += 1;
        }
    }

    /// Add one read's minimizers to the index (second pass).
    ///
    /// Minimizers with occurrence count < `min_abund` are silently dropped.
    pub fn push(&mut self, read_id: u32, minimizers: &[Minimizer]) -> Result<()> {
        self.n_reads_total += 1;
        for m in minimizers {
            let count = self.counts.get(&m.hash).copied().unwrap_or(0);
            if count >= self.min_abund {
                self.batch.push(entry_bytes(m.hash, read_id));
            }
        }
        if self.batch.len() >= self.batch_size {
            self.flush_batch()?;
        }
        self.shard.tick(4)?;
        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }
        self.batch.par_sort_unstable();
        self.batch.dedup(); // a hash may appear at multiple positions in one read
        flush_sorted_batch(
            &self.shard.env,
            self.min_db,
            &self.batch,
            WriteFlags::empty(),
        )?;
        self.batch.clear();
        Ok(())
    }

    /// Flush and write metadata.
    pub fn finalize(mut self) -> Result<()> {
        self.flush_batch()?;
        write_meta(
            &self.shard.env,
            self.meta_db,
            &[
                (b"n_reads".as_ref(), &self.n_reads_total.to_le_bytes()),
                (b"l".as_ref(), &self.l.to_le_bytes()),
                (b"read_id_width".as_ref(), &4u32.to_le_bytes()),
            ],
        )?;
        self.shard.finalize()
    }

    /// Return a reference to the accumulated occurrence counts for use in
    /// building bloom filters.
    pub fn counts(&self) -> &HashMap<u64, u32> {
        &self.counts
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimizers::Minimizer;
    use super::super::lmdb::{open_env, TEST_MAP_SIZE};
    use lmdb::{Cursor, DatabaseFlags, Transaction};
    use tempfile::TempDir;

    fn make_min(hash: u64) -> Minimizer {
        Minimizer { hash, pos: 0 }
    }

    #[test]
    fn minabund_filters_singletons() {
        let dir = TempDir::new().unwrap();
        let prefix = dir.path().join("mi");
        let mut writer = MinimIndexWriter::new(&prefix, 100, 0, 11, 2).unwrap();

        // hash 42 appears in 2 reads → kept; hash 99 appears in 1 → dropped.
        let r1 = vec![make_min(42), make_min(99)];
        let r2 = vec![make_min(42)];
        writer.count(&r1);
        writer.count(&r2);
        writer.push(1, &r1).unwrap();
        writer.push(2, &r2).unwrap();
        writer.finalize().unwrap();

        let shard_path = dir.path().join("mi_shard_0.lmdb");
        let env = open_env(&shard_path, TEST_MAP_SIZE, 4).unwrap();
        let db = env
            .open_db(Some("minimizers"))
            .unwrap();
        let txn = env.begin_ro_txn().unwrap();

        // hash 42 should be present with 2 read IDs.
        let key = 42u64.to_le_bytes();
        let mut cursor = txn.open_ro_cursor(db).unwrap();
        let entries: Vec<(u64, u32)> = cursor
            .iter_dup_of(&key)
            .unwrap()
            .map(|(_, v): (&[u8], &[u8])| {
                let val = u32::from_le_bytes(v.try_into().unwrap());
                (42u64, val)
            })
            .collect();
        assert_eq!(entries.len(), 2);

        // hash 99 should NOT be present.
        let key99 = 99u64.to_le_bytes();
        assert!(txn.get(db, &key99).is_err());
    }
}
