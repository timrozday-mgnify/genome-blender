//! Low-level LMDB helpers: opening environments, batch flushing with APPEND
//! optimisation, and shard-list management.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};

/// LMDB map size: 128 GiB virtual address space (sparse on most filesystems).
pub const MAP_SIZE: usize = 128 * 1024 * 1024 * 1024;

/// Smaller map for use in tests (64 MiB).
#[cfg(test)]
pub const TEST_MAP_SIZE: usize = 64 * 1024 * 1024;

/// MDB cursor operation: position at the last key.
const MDB_LAST: u32 = 6;

/// Open (or create) an LMDB environment at `path` with `max_dbs` sub-databases.
pub fn open_env(path: &Path, map_size: usize, max_dbs: u32) -> Result<Environment> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("creating LMDB dir {:?}", path))?;
    Environment::new()
        .set_map_size(map_size)
        .set_max_dbs(max_dbs)
        .open(path)
        .with_context(|| format!("opening LMDB at {:?}", path))
}

/// Create or open a named sub-database.
pub fn open_db(env: &Environment, name: &str, flags: DatabaseFlags) -> Result<Database> {
    env.create_db(Some(name), flags)
        .with_context(|| format!("opening sub-database {:?}", name))
}

/// Read the last key stored in `db`. Returns `None` if the database is empty.
pub fn last_key(env: &Environment, db: Database) -> Result<Option<Vec<u8>>> {
    let txn = env.begin_ro_txn()?;
    let cursor = txn.open_ro_cursor(db)?;
    match cursor.get(None, None, MDB_LAST) {
        Ok((Some(k), _)) => Ok(Some(k.to_vec())),
        Ok((None, _)) => Ok(None),
        Err(lmdb::Error::NotFound) => Ok(None),
        Err(e) => Err(e).context("last_key cursor"),
    }
}

/// Flush a pre-sorted batch of `(key_bytes, value_bytes)` pairs to `db`.
///
/// Records with `key > last_existing_key` are written with `WriteFlags::APPEND`
/// for O(1) inserts; those with `key ≤ last_existing_key` use a regular put
/// (necessary when a key already exists or when a new shard is being populated
/// from keys that overlap with a previous shard).
///
/// `extra_flags` is ORed into both write variants (e.g. `APPEND_DUP` for
/// DUPSORT databases).
///
/// The batch **must** be sorted by key bytes in ascending lexicographic order
/// before calling this function.
pub fn flush_sorted_batch<K, V>(
    env: &Environment,
    db: Database,
    batch: &[(K, V)],
    _extra_flags: WriteFlags,
) -> Result<()>
where
    K: AsRef<[u8]>,
    V: AsRef<[u8]>,
{
    if batch.is_empty() {
        return Ok(());
    }

    let last = last_key(env, db)?;

    // Partition: records with key <= last_key need regular puts;
    // records with key > last_key can use APPEND.
    let append_start = match &last {
        None => 0,
        Some(lk) => batch.partition_point(|(k, _)| k.as_ref() <= lk.as_slice()),
    };

    // Track the last key for which we used APPEND. MDB_APPEND requires the new
    // key to be STRICTLY greater than the last key, so DUPSORT duplicate entries
    // for the same key must not use APPEND (use empty flags instead — LMDB
    // inserts new dups in sorted position without any ordering constraint).
    //
    // APPEND_DUP is intentionally avoided: it requires the new dup to be
    // strictly greater than the last-written dup for that key, which is only
    // guaranteed within a single flush but not across flushes (a later read's
    // smaller read_id may land in the overlap region and must be inserted freely).
    let mut last_append_key: Vec<u8> = last.unwrap_or_default();

    let mut txn = env.begin_rw_txn()?;
    for (k, v) in &batch[..append_start] {
        txn.put(db, &k.as_ref(), &v.as_ref(), WriteFlags::empty())?;
    }
    for (k, v) in &batch[append_start..] {
        let key = k.as_ref();
        let flags = if key != last_append_key.as_slice() {
            last_append_key.clear();
            last_append_key.extend_from_slice(key);
            WriteFlags::APPEND
        } else {
            WriteFlags::empty()
        };
        txn.put(db, &key, &v.as_ref(), flags)?;
    }
    txn.commit()?;
    Ok(())
}

/// Write `key → value` pairs to the `"meta"` sub-database.
pub fn write_meta(env: &Environment, db: Database, entries: &[(&[u8], &[u8])]) -> Result<()> {
    let mut txn = env.begin_rw_txn()?;
    for (k, v) in entries {
        txn.put(db, k, v, WriteFlags::empty())?;
    }
    txn.commit()?;
    Ok(())
}

// ── Shard management ─────────────────────────────────────────────────────────

/// Tracks shard LMDB files for one index type (reads, minimizer, or combo).
pub struct ShardWriter {
    prefix: PathBuf,
    suffix: &'static str,
    shard_list_suffix: &'static str,
    shards: Vec<PathBuf>,
    shard_size: u64,
    reads_in_shard: u64,
    pub env: Environment,
    pub map_size: usize,
}

impl ShardWriter {
    /// Create a new `ShardWriter`.
    ///
    /// `suffix` is appended after `_shard_N` to form the directory name
    /// (e.g. `".lmdb"` → `{prefix}_shard_0.lmdb`).
    pub fn new(
        prefix: &Path,
        suffix: &'static str,
        shard_list_suffix: &'static str,
        shard_size: u64,
        map_size: usize,
        max_dbs: u32,
    ) -> Result<Self> {
        let path = shard_path(prefix, suffix, 0);
        let env = open_env(&path, map_size, max_dbs)?;
        Ok(ShardWriter {
            prefix: prefix.to_owned(),
            suffix,
            shard_list_suffix,
            shards: vec![path],
            shard_size,
            reads_in_shard: 0,
            env,
            map_size,
        })
    }

    /// Increment the read counter and rotate to a new shard if the threshold
    /// has been reached. Returns `true` if a rotation occurred.
    ///
    /// Caller must flush any pending batch to `self.env` BEFORE calling this.
    pub fn tick(&mut self, max_dbs: u32) -> Result<bool> {
        self.reads_in_shard += 1;
        if self.shard_size > 0 && self.reads_in_shard >= self.shard_size {
            self.rotate(max_dbs)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn rotate(&mut self, max_dbs: u32) -> Result<()> {
        self.reads_in_shard = 0;
        let idx = self.shards.len();
        let path = shard_path(&self.prefix, self.suffix, idx);
        self.env = open_env(&path, self.map_size, max_dbs)?;
        self.shards.push(path);
        Ok(())
    }

    /// Write the shard-list file listing all shard paths, one per line.
    pub fn finalize(&self) -> Result<()> {
        let list_path = self
            .prefix
            .with_extension("") // strip any extension first
            .with_file_name(format!(
                "{}{}",
                self.prefix.file_name().unwrap_or_default().to_string_lossy(),
                self.shard_list_suffix
            ));
        let content: String = self
            .shards
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&list_path, content)
            .with_context(|| format!("writing shard list {:?}", list_path))
    }
}

fn shard_path(prefix: &Path, suffix: &str, idx: usize) -> PathBuf {
    // e.g.  /data/out_shard_0.lmdb
    let filename = format!(
        "{}_shard_{idx}{suffix}",
        prefix.file_name().unwrap_or_default().to_string_lossy()
    );
    prefix.parent().unwrap_or(Path::new(".")).join(filename)
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_env(dir: &TempDir, name: &str) -> (Environment, Database) {
        let path = dir.path().join(name);
        let env = open_env(&path, TEST_MAP_SIZE, 4).unwrap();
        let db = open_db(&env, "data", DatabaseFlags::empty()).unwrap();
        (env, db)
    }

    #[test]
    fn last_key_empty_db_is_none() {
        let dir = TempDir::new().unwrap();
        let (env, db) = temp_env(&dir, "e1");
        assert!(last_key(&env, db).unwrap().is_none());
    }

    #[test]
    fn flush_and_last_key() {
        let dir = TempDir::new().unwrap();
        let (env, db) = temp_env(&dir, "e2");
        let batch: Vec<([u8; 4], [u8; 4])> = vec![
            (1u32.to_le_bytes(), 10u32.to_le_bytes()),
            (2u32.to_le_bytes(), 20u32.to_le_bytes()),
            (3u32.to_le_bytes(), 30u32.to_le_bytes()),
        ];
        flush_sorted_batch(&env, db, &batch, WriteFlags::empty()).unwrap();
        // last key is 3 (LE = [3,0,0,0])
        let lk = last_key(&env, db).unwrap().unwrap();
        assert_eq!(lk, &[3, 0, 0, 0]);
    }

    #[test]
    fn flush_multiple_batches_append() {
        let dir = TempDir::new().unwrap();
        let (env, db) = temp_env(&dir, "e3");
        // First batch: IDs 1..5
        let batch1: Vec<([u8; 4], [u8; 4])> = (1u32..=5)
            .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
            .collect();
        flush_sorted_batch(&env, db, &batch1, WriteFlags::empty()).unwrap();
        // Second batch: IDs 6..10 (all after last key → all APPENDed)
        let batch2: Vec<([u8; 4], [u8; 4])> = (6u32..=10)
            .map(|i| (i.to_le_bytes(), i.to_le_bytes()))
            .collect();
        flush_sorted_batch(&env, db, &batch2, WriteFlags::empty()).unwrap();

        // Verify all 10 entries present
        let txn = env.begin_ro_txn().unwrap();
        for i in 1u32..=10 {
            let val = txn.get(db, &i.to_le_bytes()).unwrap();
            assert_eq!(val, &i.to_le_bytes());
        }
    }

    #[test]
    fn shard_writer_creates_shard_list() {
        let dir = TempDir::new().unwrap();
        let prefix = dir.path().join("out");
        let sw =
            ShardWriter::new(&prefix, ".lmdb", "_shard_list", 0, TEST_MAP_SIZE, 4).unwrap();
        sw.finalize().unwrap();
        let list_path = dir.path().join("out_shard_list");
        let content = std::fs::read_to_string(list_path).unwrap();
        assert!(content.contains("out_shard_0.lmdb"));
    }
}
