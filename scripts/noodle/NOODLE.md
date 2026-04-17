# noodle

`noodle` builds LMDB indexes from paired-end or single-end FASTQ/FASTA files for use by `asf_sample.py` and related Python analysis scripts. It replaces the `--index-only` usage of `bin/rust-mdbg` with a focused, self-contained tool.

---

## Building

```bash
cd scripts/noodle
cargo build --release
cargo test
```

The binary is at `target/release/noodle`.

---

## Subcommands

### `noodle index`

Reads FASTQ/FASTA input (gzip accepted) and writes four LMDB indexes plus three bloom filter files.

```
noodle index --reads R1.fastq.gz [--reads2 R2.fastq.gz] --prefix /path/to/out [options]
```

**Required flags**

| Flag | Description |
|------|-------------|
| `--reads <FILE>` | Input FASTQ/FASTA (R1, or single-end). Gzip accepted. |
| `--prefix <PATH>` | Output prefix. All output files are named `{prefix}.{type}`. The directory must exist. |

**Minimizer parameters**

| Flag | Default | Description |
|------|---------|-------------|
| `-l <INT>` | 17 | l-mer length for minimizer extraction. |
| `--density <FLOAT>` | 0.1 | Fraction of l-mers selected as minimizers. An l-mer is selected when its canonical NtHash ≤ `density × 2^64`. |
| `--minabund <INT>` | 2 | Minimum number of reads a minimizer must appear in to be written to the minimizer index. |

**Paired-end**

| Flag | Default | Description |
|------|---------|-------------|
| `--reads2 <FILE>` | — | R2 file for paired-end mode. |
| `--pe-combo-density <FLOAT>` | 0.05 | Fraction of cross-pair combo hashes to retain. |
| `--combo-max-distance <INT>` | 500 | Maximum position distance (in original-sequence coordinates) between an R1 and R2 minimizer for a PE combo to be recorded. |

**Intra-read combos**

| Flag | Default | Description |
|------|---------|-------------|
| `--intra-combo-density <FLOAT>` | 0.05 | Fraction of within-read minimizer pair hashes to retain. |

**Performance tuning**

| Flag | Default | Description |
|------|---------|-------------|
| `--reads-batch-size <INT>` | 100 000 | Reads accumulated in memory before a LMDB flush for the reads index. |
| `--minimizer-batch-size <INT>` | 500 000 | Same for the minimizer index. |
| `--combo-batch-size <INT>` | 500 000 | Same for combo indexes. |
| `--reads-shard-size <INT>` | 0 | Reads per LMDB shard file for the reads index (0 = single file). |
| `--minimizer-shard-size <INT>` | 0 | Same for the minimizer index. |
| `--combo-shard-size <INT>` | 0 | Same for combo indexes. |

### `noodle view`

Prints index contents to stdout as tab-separated values. Useful for debugging and manual inspection.

```
noodle view reads        <prefix>   # read_id<TAB>hex_hashes (space-separated)
noodle view minimizers   <prefix>   # minimizer_hash<TAB>read_id
noodle view pe-combo     <prefix>   # combo_hash<TAB>count
noodle view intra-combo  <prefix>   # combo_hash<TAB>count
```

All hashes are printed in hexadecimal (`0x…`). The `reads` subcommand prints one row per read; `minimizers` prints one row per (hash, read_id) pair across all shards.

---

## Output files

Given `--prefix /data/out`, `noodle index` produces the following files.

| File | Description |
|------|-------------|
| `out.index_shard_0.lmdb/` | Reads index (first shard; more if `--reads-shard-size > 0`) |
| `out.index_shard_list` | Text file listing all reads index shard paths, one per line |
| `out.minimizer_index_shard_0.lmdb/` | Minimizer index (first shard) |
| `out.minimizer_index_shard_list` | Shard list for minimizer index |
| `out.intra_combo_shard_0.lmdb/` | Intra-read combo index (first shard) |
| `out.intra_combo_shard_list` | Shard list for intra-combo index |
| `out.pe_combo_shard_0.lmdb/` | PE combo index (first shard; empty/absent for single-end) |
| `out.pe_combo_shard_list` | Shard list for PE combo index |
| `out.minimizer_bloom.bin` | Bloom filter for minimizer index |
| `out.pe_combo_bloom.bin` | Bloom filter for PE combo index |
| `out.intra_combo_bloom.bin` | Bloom filter for intra-combo index |

Each `.lmdb` entry is an LMDB *environment directory* (not a single file). LMDB creates `data.mdb` and `lock.mdb` inside it.

---

## Index data structures

### Reads index (`out.index_shard_*.lmdb`)

Maps every read to its ordered sequence of minimizer hashes.

**`"reads"` sub-database** — regular (no DUPSORT):

```
key   = read_id          u32 little-endian, 4 bytes
value = minimizer hashes packed u64 little-endian, 8 bytes × n_minimizers
```

Minimizer hashes are stored in the order they appear along the read (left to right). A read with zero minimizers (shorter than `l` after HPC) has no entry.

**`"meta"` sub-database:**

```
"n_reads"       → u64 LE  (reads written to this shard)
"l"             → u32 LE  (l-mer length)
"read_id_width" → u32 LE = 4
```

**Read ID numbering:**

- Single-end: sequential 1-based IDs — read 1, 2, 3, …
- Paired-end: R1 reads get odd IDs (1, 3, 5, …); R2 reads get even IDs (2, 4, 6, …). The paired mate of read ID `n` is `n+1` (R1→R2) or `n-1` (R2→R1).

### Minimizer index (`out.minimizer_index_shard_*.lmdb`)

Inverted map: for each minimizer hash that appears in ≥ `minabund` reads, stores every read ID that contains it.

**`"minimizers"` sub-database** — DUPSORT + DUPFIXED:

```
key   = minimizer_hash   u64 little-endian, 8 bytes
value = read_id          u32 little-endian, 4 bytes  (one dup entry per read)
```

The DUPSORT flag means each key can have multiple sorted values. The DUPFIXED flag declares all values the same size (4 bytes), which allows LMDB to pack them more efficiently.

Only minimizers whose total occurrence count across all reads (including both R1 and R2 in paired-end mode) meets `--minabund` are written. Singletons are dropped.

**`"meta"` sub-database:** same schema as the reads index.

### Combo indexes (`out.{pe,intra}_combo_shard_*.lmdb`)

Both PE and intra-read combo indexes share the same schema.

**`"combo"` sub-database** — regular:

```
key   = combo_hash   u64 little-endian, 8 bytes
value = count        u32 little-endian, 4 bytes
```

`count` is the number of times this combo hash was observed across all reads/pairs.

**`"meta"` sub-database:**

```
"n_reads"  → u64 LE  (reads/pairs contributing to this shard)
"l"        → u64 LE  (l-mer length)
"density"  → f64 LE  (combo density threshold, stored as raw bit pattern)
```

**Combo hash function:**

The hash of a minimizer pair `(a, b)` is commutative (`combo_hash(a,b) == combo_hash(b,a)`):

```rust
fn combo_hash(a: u64, b: u64) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let h = lo ^ hi.wrapping_mul(0x9e3779b97f4a7c15);
    let h = (h ^ (h >> 30)).wrapping_mul(0x6c62272e07bb0142);
    let h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}
```

This is a splitmix64-derived mixing function. Commutativity means the same hash is produced whether the pair came from (R1, R2) or (R2, R1), and within a read the ordering of the two minimizers doesn't matter.

**PE combos:** all (R1 minimizer, R2 minimizer) pairs where `|pos_R1 − pos_R2| ≤ combo_max_distance`. The position is in original (non-HPC) sequence coordinates.

**Intra-read combos:** all C(n, 2) pairs of minimizers within a single read.

A combo hash is retained only if its value ≤ `density × 2^64`, applying the same thinning used for minimizers.

### Bloom filters

Three binary files, one per index type. All use the same 40-byte header:

```
Offset  Bytes  Field
0        8     Magic (see table below)
8        8     Version (u64 LE = 1)
16       8     Extra: l-mer length as u64 LE (minimizer bloom) or combo density as raw f64 bits (combo blooms)
24       8     n_bits (u64 LE; always 2^30 = 1 073 741 824)
32       8     n_hash_fns (u64 LE = 3)
40      128 MB Packed bit array, 64-bit words in little-endian order
```

| File | Magic | Extra field |
|------|-------|-------------|
| `minimizer_bloom.bin` | `MNBLOOMB` | l-mer length |
| `pe_combo_bloom.bin` | `PECMBBFL` | PE combo density (f64 bits) |
| `intra_combo_bloom.bin` | `ICMBBFL1` | Intra-combo density (f64 bits) |

Hash function: FNV-1a 64-bit double-hashing. For a value `v`, two independent hashes are derived:

```
h1 = fnv1a_64(v)
h2 = rotate_left(h1, 17) × 0x9e3779b97f4a7c15
bit_pos[i] = (h1 + i × h2) mod n_bits   for i in 0..3
```

At 2^30 bits and 3 hash functions, the theoretical false-positive rate is ≈ 1% for up to ~100 M distinct elements.

---

## Minimizer extraction

Minimizers are computed in three steps:

1. **Homopolymer compression (HPC):** runs of identical bases are collapsed — `AAACCCGGG` → `ACG`. This reduces sensitivity to homopolymer-length variation (common in short-read error profiles) and mirrors the approach used in rust-mdbg.

2. **NtHash rolling hash:** the canonical NtHash (minimum of forward and reverse-complement hashes) is computed for every l-mer in the HPC sequence. NtHash is O(1) per step, making this linear in the read length.

3. **Density thresholding:** an l-mer is selected as a minimizer if its hash ≤ `density × 2^64`. This selects approximately `density` fraction of l-mers uniformly at random (the hash is pseudorandom over the uniform distribution). Minimizer positions are mapped back to original (non-HPC) coordinates for storage.

**Choosing `l` and `density`:** a useful rule of thumb is that `density × read_length × 1.5 ≈ l`, which gives a good balance between minimizer density and uniqueness. For 150 bp Illumina reads with `l = 17`, a density of `17 / (150 × 1.5) ≈ 0.076` yields ~10–15 minimizers per read.

---

## Processing pipeline

### Two-pass design

`noodle index` reads the input FASTQ(s) twice:

**Pass 1 — count pass:** scans all reads and accumulates a `HashMap<minimizer_hash, occurrence_count>`. No LMDB writes. This pass is fast (no I/O beyond reading FASTQ).

**Pass 2 — write pass:** re-scans all reads. For each read, minimizers are written to the reads index and — if their count from pass 1 meets `--minabund` — to the minimizer index. Combo hashes are computed and accumulated for both indexes. Bloom filters are built from the accumulated counts after pass 2 completes.

### Paired-end spill file

During pass 2 in paired-end mode, noodle processes R1 and R2 in separate sub-passes (R1 first, then R2). To compute PE combo hashes — which require knowing both R1 and R2 minimizers for a pair — R1 minimizer data is written to a temporary binary spill file while processing R1. When R2 is processed, the corresponding R1 entry is read back from the spill file. The spill file is deleted on completion.

Spill record format per read:

```
n         u32 LE  (number of minimizers)
hash[0]   u64 LE  |
pos[0]    u32 LE  | repeated n times
hash[i]   u64 LE  |
pos[i]    u32 LE  |
```

### LMDB batch flushing

All four writers accumulate records in memory up to their batch size, then flush to LMDB in one transaction. Before each flush:

1. Batch is sorted by key bytes (ascending lexicographic).
2. The current maximum key in the database is read.
3. Records with key ≤ max key are written with `WriteFlags::empty()` (ordinary put).
4. Records with key > max key are written with `WriteFlags::APPEND` (O(1) B-tree insert rather than O(log N)).

For DUPSORT databases (minimizer index), a key that matches the last-written key in the append region must use `WriteFlags::empty()` rather than `APPEND` — LMDB requires strictly-greater keys for APPEND, even in DUPSORT mode.

The combo indexes additionally perform a read-modify-write for keys in the overlap region: the existing count is read, the delta is added, and the sum is written back.

---

## Sharding

When `--reads-shard-size N` (or the equivalent for minimizer/combo) is greater than zero, a new LMDB environment is created every N reads. Shard files are named:

```
{prefix}.{type}_shard_0.lmdb
{prefix}.{type}_shard_1.lmdb
…
```

A companion shard-list file (`{prefix}.{type}_shard_list`) is written containing the absolute paths of all shards, one per line. The Python scripts read this file to discover all shards.

Sharding is useful for very large datasets where a single LMDB file would exceed the available virtual address space or become unwieldy. The default is no sharding (single file per index).

---

## Comparison with rust-mdbg

`noodle` was developed as a replacement for the `--index-only` invocation of `bin/rust-mdbg`. The key differences:

### What rust-mdbg did in `--index-only` mode

`rust-mdbg --index-only --dump-minimizer-index …` skips graph construction, but the minabund filtering in rust-mdbg is implemented through the `minimizer_to_int` lookup table, which is populated during graph construction. With `--index-only` and no external `--lmer-counts` file, `has_lmer_counts = false` and the lookup is skipped — **all density-selected minimizers are written to the minimizer index regardless of `--minabund`**. The `--minabund` flag had no effect on the index.

### What noodle does differently

| Property | noodle | rust-mdbg (`--index-only`) |
|----------|--------|---------------------------|
| FASTQ passes | 2 (count then write) | 1 |
| minabund filtering in LMDB | Exact (two-pass count) | No filtering |
| Minimizer LMDB size | Smaller (singletons excluded) | Larger |
| Bloom pre-filter in asf_sample | Yes | Yes |
| Graph construction | None | None |
| PE combo index | Yes | Yes |
| Intra combo index | Yes | Yes |

The extra FASTQ pass is inexpensive compared to writing LMDB, and the resulting minimizer index is smaller and cleaner. `asf_sample.py` applies bloom-filter pre-filtering before any LMDB query, so even if singleton minimizers were present they would mostly be rejected — but excluding them at write time reduces index size and speeds up lookups.

### Minimizer hash space

rust-mdbg uses NtHash with HPC compression in the same way noodle does. The canonical hash (min of forward and RC) is identical between the two tools for the same `l` and `density` parameters, so indexes built by noodle are interchangeable with those built by rust-mdbg (modulo the minabund filtering difference above).

### Removed functionality

noodle does not implement:
- Graph construction or GFA output
- k-min-mer (k-mer of minimizers) index — `asf_sample.py` constructs k-min-mers in Python from the reads index
- Succession index
- Error correction
- POA alignment or base-space conversion
