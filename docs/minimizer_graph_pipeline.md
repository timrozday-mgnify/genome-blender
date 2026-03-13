# Minimizer-Space De Bruijn Graph Pipeline

Technical reference for the rust-mdbg assembly pipeline and the `scripts/parse_gfa.py`
analysis framework. Covers how reads are transformed into minimizers, how the de Bruijn
graph is built and serialised, how paths are sampled, and how reads are mapped back to
paths to estimate insert sizes. Paired-end handling and reverse-complement awareness are
traced throughout.

---

## 1. rust-mdbg: from reads to graph

### 1.1 Minimizer extraction

Every input read is processed to produce an ordered sequence of *minimizer IDs* — u64
NT-hash values derived from short sub-sequences called *l-mers*.

**NT-hash** (`nthash::ntc64`) is a rolling hash function for nucleotide sequences.
For a given l-mer it returns a single u64 that is the same regardless of whether the
l-mer or its reverse complement is supplied (canonical hashing). This is the basis of
reverse-complement awareness.

**Density-based selection** (`MINIMIZER_TYPE = Density`):

1. The read is homopolymer-compressed (HPC): consecutive identical bases are collapsed
   to one. Positions in the original read that correspond to each HPC position are
   retained.
2. Every l-mer in the HPC read is hashed with NT-hash.
3. A hash is selected as a minimizer if `hash ≤ density × u64::MAX`.
   `density` (0–1) is a command-line parameter; lower density = fewer minimizers.
4. The resulting sequence of selected hashes, and their positions in the original
   (pre-HPC) read, is stored in the `Read` struct as `transformed: Vec<u64>` and
   `minimizers_pos: Vec<usize>`.

**Syncmer-based selection** is an alternative that uses canonical s-mers in a sliding
window to decide whether each l-mer is selected; the same NT-hash is computed for the
selected l-mer.

**Reverse-complement awareness (`REVCOMP_AWARE = true`):**

NT-hash is symmetric: `ntc64(lmer) == ntc64(revcomp(lmer))`. Because of this, a
minimizer selected from the forward strand and the same minimizer selected from the
reverse strand produce the same hash ID. The graph therefore represents both strands
with the same node. Which orientation was *traversed* is tracked separately (see §1.2).

**Minimizer table** (`{prefix}.minimizer_table`): a plain-text TSV file written at the
end of the run mapping each selected u64 hash to the l-mer string that produced it.
Format: `{hash_u64}<TAB>{lmer_string}`, one entry per line. Because NT-hash is
canonical, both a forward l-mer and its reverse complement resolve to the same hash and
therefore to the same row.

### 1.2 K-mer formation and canonicalisation

A *k-mer in minimizer space* is a vector of `k` consecutive minimizer IDs:
`[transformed[i], transformed[i+1], ..., transformed[i+k-1]]`.

Each k-mer is **canonicalised**: the k-mer and its reverse (element-wise reversal of
the ID vector, equivalent to the reverse-complement strand) are compared and the
lexicographically smaller is stored. A boolean flag records whether the stored form is
the reverse of what was observed. This flag propagates into the GFA link orientations.

K-mers are counted across all reads. Nodes with abundance below `--minabund` are
discarded.

### 1.3 Graph construction

**Nodes** are unique canonical k-mers. Each node stores:
- The k minimizer IDs (in canonical order).
- The DNA sequence spanning those minimizers (from the first selected l-mer to the
  last, including the bases between them).
- Abundance (how many times the k-mer was observed).

**Edges** connect any two nodes whose k-mers overlap in `k − 1` minimizers (sliding
window by one position). Because nodes are canonical, edges carry orientation
information: each L-record in the GFA encodes the orientations of the two endpoints
at the time the overlap was observed.

### 1.4 GFA output

The graph is written as a GFA 1.0 file.

**Header:**
```
H	VN:Z:1.0
```

**Segment (S) records** — one per node:
```
S	{node_index}	*	LN:i:{bp_length}	KC:i:{abundance}
```
- `node_index`: sequential integer, used as the segment name.
- Sequence field: always `*` (bases are in the `.sequences` files).
- `LN:i:`: total base-pair length of the node's sequence.
- `KC:i:`: k-mer abundance.

**Link (L) records** — one per edge:
```
L	{from_idx}	{from_orient}	{to_idx}	{to_orient}	{overlap}M
```
- `from_orient`, `to_orient`: `+` or `−`. These encode which canonical orientation of
  each node was on the same strand as the observed overlap.
- `overlap`: approximate overlap length in base pairs.

### 1.5 .sequences files

Written per thread; pattern `{prefix}.{thread}.sequences`. LZ4-compressed TSV.

**Header comments** (appear once per file):
```
# k = {k_value}
# l = {l_value}
# Structure of remaining of the file:
# [node name]	[list of minimizers]	[sequence of node]	[abundance]	[origin]	[shift]
```

**Data lines** (one per node, tab-separated):
```
{node_index}	[{hash1}, {hash2}, ..., {hash_k}]	{dna_sequence}	{abundance}	*	({shift0}, {shift1})
```

- Node name matches the GFA S-record index exactly.
- Minimizer list: bracket-enclosed comma-separated u64 NT-hash values *in canonical
  (stored) order*. About half of nodes are in the reverse-complement orientation
  relative to the genomic direction in which they were first observed (because
  `REVCOMP_AWARE` always stores the lexicographically smaller form).
- `shift`: a 2-tuple `(offset_of_second_minimizer, offset_of_second_to_last_minimizer)`
  used to compute base-pair overlaps between adjacent nodes.

### 1.6 .read_minimizers files

Written per thread when `--dump-read-minimizers` is supplied; pattern
`{prefix}.{thread}.read_minimizers`. LZ4-compressed.

**Binary format** (detected by magic bytes):

```
Bytes 0–3:  b"RMBG"          — magic
Byte  4:    0x01              — version

Per read (repeated until EOF):
  4 bytes LE u32   name_len
  name_len bytes   read name (UTF-8)
  4 bytes LE u32   n          — number of minimizers
  n × 8 bytes      minimizer_ids (LE u64 array)
  n × 8 bytes      positions   (LE u64 array, positions in original read)
```

**Legacy TSV format** (files without the magic header):
```
# comment lines
{read_name}<TAB>{comma-separated minimizer IDs}<TAB>{comma-separated positions}
```

The minimizer IDs stored here are the *same u64 NT-hash values* as in the node
minimizer lists. A read's IDs appear in the order they were selected along the read,
which is the *observed* order — not necessarily the canonical order of any node.

### 1.7 Paired-end reads in rust-mdbg

rust-mdbg does not model paired-end reads internally. R1 and R2 are treated as
independent reads. Insert size estimation is performed entirely in `parse_gfa.py`
(§3.5) by detecting mate pairs from read names and measuring the distance between
their positions on sampled paths.

---

## 2. Reverse-complement handling: the key invariant

The central invariant of REVCOMP_AWARE minimizer de Bruijn graphs:

> **A node's minimizer list is stored in one canonical orientation. Approximately
> half of all nodes are stored in the reverse-complement orientation relative to
> the genomic direction in which a read traverses them.**

Consequences:

1. When assembling a path through the graph, each node must be consulted in either
   its stored orientation (`+`) or its reverse (`−`) depending on which strand the
   traversal is on.
2. When matching a read against a path, the read's minimizer sequence may appear
   forward *or* reversed in the path, depending on which orientation the nodes were
   stored in.
3. Both orientations of every read must be searchable.

The GFA L-records encode, for each edge, the orientations of the two endpoint nodes
at the time the overlap was observed. `parse_gfa.py` reconstructs the traversal
orientation of each step from these records (§3.3).

---

## 3. parse_gfa.py: loading, sampling, and matching

### 3.1 GFA parsing

`parse_gfa(path)` reads the GFA line by line and builds an undirected `rx.PyGraph`
(rustworkx):
- S-records → nodes with `Segment` data (`name`, `length`, `kmer_count`).
- L-records → edges with `Link` data (`from_orient`, `to_orient`, `overlap`,
  `from_idx`, `to_idx`).

The `from_idx` / `to_idx` fields on `Link` store the internal graph node indices so
that edge traversal can resolve orientations without a name lookup.

### 3.2 Loading segment minimizers

`load_segment_minimizers(prefix)` reads all `{prefix}.*.sequences` files.

For each segment:
- The bracket-enclosed minimizer list is parsed into a 1-D `np.ndarray` of dtype
  `uint64` (`fwd`).
- The reversed copy is precomputed once: `rev = fwd[::-1].copy()`.
- Both are stored: `seg_min[node_name] = (fwd, rev)`.

`build_seg_min_index(graph, seg_min)` converts the name-keyed dict into a list indexed
by graph node index for O(1) lookup during path construction.

### 3.3 Orientation-aware path sampling

**_OrientedPath** = `list[tuple[int, bool]]` — each element is a
`(node_idx, is_forward)` pair. `is_forward=True` means the node's stored minimizer
sequence is used as-is; `False` means it is reversed.

**_next_orient(cur_idx, cur_orient, link)** decodes the GFA L-record:
- If `cur_idx == link.from_idx` and `cur_orient == link.from_orient`:
  next orientation = `link.to_orient`.
- If `cur_idx == link.to_idx` and `cur_orient == opp(link.to_orient)`:
  next orientation = `opp(link.from_orient)`.
- Otherwise: edge not traversable from this state; return `None`.

This correctly propagates orientation through each step of the walk.

**_random_simple_path** starts from a leaf node (degree 1), reads the start
orientation from the single incident edge, then greedily extends to unvisited
neighbours. Neighbour selection is weighted by k-mer count, overlap length, or
uniformly, depending on `--weight`.

### 3.4 Path minimizer sequence construction

`path_minimizer_sequence(path, seg_min_index, k)` assembles a single uint64 array
representing the minimizer content of the path in traversal order.

Adjacent segments in a k-mer de Bruijn graph share exactly `k − 1` minimizers (the
overlap between consecutive k-mers). The overlapping prefix of each segment after the
first is therefore dropped:

```
parts = []
for node_idx, is_forward in path:
    arr = seg_min_index[node_idx][0 if is_forward else 1]
    parts.append(arr if not parts else arr[k-1:])
return np.concatenate(parts)
```

The `not parts` guard (rather than `i == 0`) correctly handles paths where early nodes
have no minimizer data — the first *contributing* node is never trimmed.

### 3.5 Read index and pair detection

`load_read_minimizers(prefix)` reads all `{prefix}.*.read_minimizers` files, returning
`{read_name: tuple_of_minimizer_ids}`. Both binary (magic `b"RMBG\x01"`) and legacy
TSV formats are supported, auto-detected per file.

`build_read_index(read_minimizers)` assigns a compact integer ID to each read (sorted
by name for reproducibility) and detects paired-end mates:

Paired-end suffixes recognised (via `_PAIR_SUFFIX_RE`):

| Pattern | Examples |
|---------|----------|
| `/R?[12]$` | `/1`, `/2`, `/R1`, `/R2` |
| `_R[12]$` | `_R1`, `_R2` |
| `\.R?[12]$` | `.1`, `.2`, `.R1`, `.R2` |
| `[ \t][12]:[^ \t]+$` | CASAVA: ` 1:N:0:BARCODE` |

Bare `_1`/`_2` are intentionally excluded — indistinguishable from accession suffixes.

The stripped name (*template name*) is used as a key. Templates with exactly two
reads are cross-linked in `ReadIndex.pairs: dict[int, int]`.

`_pair_number(name)` extracts the mate number (`"1"`, `"2"`, or `"."` for unpaired)
from a read name using the same suffix patterns.

### 3.6 Building the Aho-Corasick automaton

`build_aho_corasick(index, path_minimizer_set)` builds an Aho-Corasick automaton over
integer (u64) sequences.

**Why both orientations?** Because `REVCOMP_AWARE` means a read's minimizer sequence
may appear *reversed* in the path sequence if the nodes that read traversed were
stored in reverse orientation. Both `minimizers` (forward) and `minimizers[::-1]`
(reversed) are indexed under the same `read_id`, so a match of either orientation is
attributed to the same read.

**Optional pre-filter** (`--ac-prefilter`, disabled by default): if
`path_minimizer_set` is provided, a read is only added to the automaton if every one
of its minimizer IDs appears somewhere in the union of all sampled path sequences.
This is a necessary condition for a match.

**Pair coherence**: when the pre-filter is active, both mates of a pair must
individually pass or neither is added. A lone mate in the automaton can never produce
a paired insert-size observation, and would silently suppress its partner's matches.

### 3.7 Read-to-path matching and insert size estimation

`match_reads_to_path(path_min_seq, ac)` converts the path's uint64 array to a tuple
and runs `ac.search()`, yielding `PathMatch(read_id, path_start, path_end)` for every
occurrence of any read pattern in the path.

`_path_pair_insert_sizes(path_matches, index)`:

1. Groups hit positions by `read_id`.
2. For every R1 hit, checks whether its mate R2 also hit the same path.
3. For each (R1 hit, R2 hit) combination computes the minimizer-space span:
   `max(r1_end, r2_end) − min(r1_start, r2_start)`.
4. Returns `(r1_id, r2_id, span)` tuples.

Multiple hits per read (when a read matches more than once in the path) produce
multiple span values — all Cartesian-product permutations are reported.

**Base-pair conversion:** `_minimizer_to_bp_scale` computes the mean bp-per-minimizer
ratio across all segments that have minimizer data. The ratio is applied to
minimizer-space spans to obtain base-pair insert size estimates.

### 3.8 Component filtering

`large_component_nodes(graph, min_proportion)` restricts path sampling to the largest
connected components:

1. All components are identified with `rx.connected_components()`.
2. Sorted by *total base-pair span* (sum of segment lengths) descending.
3. Components are greedily selected largest-first until the cumulative node count
   covers at least `min_proportion × total_nodes`.
4. Returns the union of selected node indices (used to filter leaf-node candidates
   before path sampling) and the ordered list of selected components (for reporting).

---

## 4. End-to-end data flow

```
FASTQ/FASTA input
       │
       ▼
rust-mdbg: extract l-mers, compute NT-hash, apply density filter
       │  minimizer IDs (u64, canonical via NT-hash)
       ▼
Form k-mers of minimizers, canonicalise, count
       │  canonical k-mers with bool orientation flag
       ▼
Build de Bruijn graph (nodes = k-mers, edges = k−1 overlap)
       │
       ├──▶  {prefix}.gfa              — graph topology (S + L records)
       ├──▶  {prefix}.*.sequences      — per-node minimizer lists + DNA (LZ4-TSV)
       ├──▶  {prefix}.*.read_minimizers — per-read minimizer IDs (LZ4-binary)
       └──▶  {prefix}.minimizer_table  — hash → l-mer string (plain TSV)
                    │
                    ▼
parse_gfa.py:
  parse_gfa()               — GFA → rustworkx PyGraph
  load_segment_minimizers() — .sequences → {seg_name: (fwd_array, rev_array)}
  build_seg_min_index()     — name dict → node-index list
  load_read_minimizers()    — .read_minimizers → {read_name: (id, id, ...)}
  build_read_index()        — detect pairs, assign integer IDs
       │
       ▼
  large_component_nodes()   — restrict sampling to largest components (optional)
  sample_paths()            — orientation-aware random walks → _OrientedPath list
       │
       ▼
  path_minimizer_sequence() — concatenate (fwd|rev) arrays with k−1 overlap drop
       │  uint64 array per path
       ▼
  build_aho_corasick()      — index all reads (fwd + rev) into AC automaton
       │
       ▼
  match_reads_to_path()     — AC search of each path sequence
  _path_pair_insert_sizes() — collect paired spans across matched reads
  _insert_size_stats()      — statistics in minimizer space and base pairs
       │
       ├──▶  stdout: graph summary, component stats, path sizes, insert sizes
       ├──▶  {prefix}.graph_summary.json       (--json)
       ├──▶  {prefix}.insert_sizes.tsv         (--insert-sizes-out)
       ├──▶  read_mappings.tsv                 (--read-mappings-out)
       └──▶  paths.tsv                         (--paths-out)
```

---

## 5. Key constants and format details

| Item | Value |
|------|-------|
| Binary read_minimizers magic | `b"RMBG\x01"` (5 bytes) |
| Binary integer endianness | Little-endian |
| Minimizer ID type | u64 (NT-hash, canonical) |
| Sequences file compression | LZ4 frame |
| Read_minimizers compression | LZ4 frame |
| Minimizer table compression | None (plain text) |
| GFA version | 1.0 |
| Node sequence in GFA | Always `*` |
| Node name type | Sequential integer (as string) |
| Overlap between adjacent nodes | `k − 1` minimizers |
| Orientation characters | `+` (stored order) / `−` (reversed) |
| Pair number separator | `.` for unpaired |
| bp/minimizer scale | `sum(seg_lengths) / sum(seg_minimizer_counts)` |
