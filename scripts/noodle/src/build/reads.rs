//! Reads LMDB index: maps each 1-based read ID to its packed minimizer hashes.
//!
//! Schema
//! ------
//! `"reads"` database (regular):
//!   key   = read ID (u32 LE, 4 bytes)
//!   value = packed u64 LE minimizer hashes (8 bytes × n_minimizers)
//!
//! `"meta"` database:
//!   `n_reads`       → u64 LE  (total reads across all shards)
//!   `l`             → u32 LE  (l-mer length)
//!   `read_id_width` → u32 LE = 4

use anyhow::Result;
use lmdb::{Database, DatabaseFlags, WriteFlags};
use rayon::prelude::*;

use super::lmdb::{flush_sorted_batch, open_db, write_meta, ShardWriter, MAP_SIZE};
use crate::minimizers::Minimizer;

/// Encodes a read ID as a 4-byte little-endian key.
#[inline]
pub fn read_id_key(id: u32) -> [u8; 4] {
    id.to_le_bytes()
}

/// Encodes a slice of minimizer hashes as a packed value byte array.
pub fn pack_minimizers(minimizers: &[Minimizer]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(minimizers.len() * 8);
    for m in minimizers {
        buf.extend_from_slice(&m.hash.to_le_bytes());
    }
    buf
}

/// In-memory batch entry for the reads index.
type ReadsBatch = Vec<([u8; 4], Vec<u8>)>;

/// Writer for the sharded reads LMDB index.
pub struct ReadsIndexWriter {
    shard: ShardWriter,
    reads_db: Database,
    meta_db: Database,
    batch: ReadsBatch,
    batch_size: usize,
    n_reads_total: u64,
    l: u32,
}

impl ReadsIndexWriter {
    pub fn new(
        prefix: &std::path::Path,
        batch_size: usize,
        shard_size: u64,
        l: u32,
    ) -> Result<Self> {
        let shard = ShardWriter::new(
            prefix,
            ".lmdb",
            "_shard_list",
            shard_size,
            MAP_SIZE,
            4,
        )?;
        let reads_db = open_db(&shard.env, "reads", DatabaseFlags::empty())?;
        let meta_db = open_db(&shard.env, "meta", DatabaseFlags::empty())?;
        Ok(ReadsIndexWriter {
            shard,
            reads_db,
            meta_db,
            batch: Vec::with_capacity(batch_size),
            batch_size,
            n_reads_total: 0,
            l,
        })
    }

    /// Add one read's minimizers. `id` is the 1-based read index.
    pub fn push(&mut self, id: u32, minimizers: &[Minimizer]) -> Result<()> {
        let key = read_id_key(id);
        let val = pack_minimizers(minimizers);
        self.batch.push((key, val));
        self.n_reads_total += 1;

        if self.batch.len() >= self.batch_size {
            self.flush_batch()?;
        }
        // Shard rotation after flushing.
        self.shard.tick(4)?;
        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }
        // Sort by LE key bytes (ascending).
        self.batch.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

        flush_sorted_batch(&self.shard.env, self.reads_db, &self.batch, WriteFlags::empty())?;
        self.batch.clear();
        Ok(())
    }

    /// Flush remaining records and write metadata. Must be called once when
    /// all reads have been pushed.
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
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimizers::Minimizer;
    use super::super::lmdb::{open_env, TEST_MAP_SIZE};
    use lmdb::{Transaction, DatabaseFlags};
    use tempfile::TempDir;

    fn make_minimizer(hash: u64) -> Minimizer {
        Minimizer { hash, pos: 0 }
    }

    #[test]
    fn pack_minimizers_correct() {
        let mins = vec![
            make_minimizer(0x0102030405060708),
            make_minimizer(0xAABBCCDDEEFF0011),
        ];
        let packed = pack_minimizers(&mins);
        assert_eq!(packed.len(), 16);
        assert_eq!(&packed[0..8], &0x0102030405060708u64.to_le_bytes());
        assert_eq!(&packed[8..16], &0xAABBCCDDEEFF0011u64.to_le_bytes());
    }

    #[test]
    fn round_trip_single_read() {
        let dir = TempDir::new().unwrap();
        let prefix = dir.path().join("reads_idx");
        let mut writer = ReadsIndexWriter::new(&prefix, 100, 0, 17).unwrap();
        let mins = vec![make_minimizer(42), make_minimizer(99)];
        writer.push(1, &mins).unwrap();
        writer.finalize().unwrap();

        // Re-open and verify.
        let shard_path = dir.path().join("reads_idx_shard_0.lmdb");
        let env = open_env(&shard_path, TEST_MAP_SIZE, 4).unwrap();
        let db = env.open_db(Some("reads")).unwrap();
        let txn = env.begin_ro_txn().unwrap();
        let val = txn.get(db, &read_id_key(1)).unwrap();
        assert_eq!(val, pack_minimizers(&mins).as_slice());
    }
}
