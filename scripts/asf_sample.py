#!/usr/bin/env python3
"""Assembly-free de Bruijn path sampling from metagenomic minimizer reads.

Samples paths through the implicit minimizer-space de Bruijn graph by
performing k−1 minimizer overlap extension directly on the indexed reads,
without constructing an assembly graph in memory.

Data sources (all produced by rust-mdbg with --dump-read-minimizers
and --dump-minimizer-index):

    {PREFIX}.index.lmdb              per-read minimizer sequences and bp positions
    {PREFIX}.minimizer_index.lmdb    inverted index: minimizer hash → read IDs
    {PREFIX}.minimizer_bloom.bin     bloom filter for fast minimizer membership checks

Algorithm (de Bruijn style, assembly-free):

    1. Seed: pick a random read and a random k-mer window within it.
    2. Extend forward: given the path-terminal k-mer tip T = [m₁…mₖ],
       look up the edge minimizer (mₖ) in the minimizer index to find reads
       that contain it, then verify the full (k−1) overlap in each read to
       collect candidate next minimizers.  If a single candidate has
       sufficient support, add it to the path and repeat.
    3. Extend backward symmetrically from the path start.
    4. Stop at dead ends (no extension), ambiguous branches (tie), or
       when the path reaches max-path-mers minimizers.

LRU caches on decoded reads and minimizer→reads lookups amortise repeated
LMDB queries across extension steps.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import mmap
import random
import struct
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Annotated, Literal

import lmdb
import numpy as np
import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

try:
    import pyro as _pyro
    import pyro.distributions as _pyro_dist
    from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDelta, AutoNormal
    from pyro.optim import Adam as _PyroAdam
    import torch as _torch
    _PYRO_AVAILABLE = True
except ImportError:
    _pyro = _pyro_dist = _torch = None  # type: ignore[assignment]
    MCMC = NUTS = Predictive = SVI = Trace_ELBO = AutoDelta = AutoNormal = _PyroAdam = None  # type: ignore[assignment]
    _PYRO_AVAILABLE = False

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

_LMDB_MAP_SIZE = 128 * 1024 * 1024 * 1024  # 128 GB
_LMDB_MAX_DBS = 8

# Combo hash mixing constants — must match rust-mdbg's canonical_combo_hash exactly.
# NOTE: CM2 here (0x6c62272e07bb0142) differs from the notebook's _CM2 constant;
# use these values so that path sketch hashes match the PE combo LMDB.
_UINT64_MAX: int = (1 << 64) - 1
_COMBO_CM1 = np.uint64(0x9E3779B97F4A7C15)
_COMBO_CM2 = np.uint64(0x6C62272E07BB0142)
_COMBO_CM3 = np.uint64(0x94D049BB133111EB)

# Default insert-size bin edges (bp).  Covers 0–20 kbp in log-ish steps.
_DEFAULT_INSERT_BINS = "0,200,400,600,800,1000,1500,2000,3000,4000,6000,8000,12000,16000,20000"

# Prior medians for Pyro models (bp).
_DEFAULT_FRAG_PRIOR_MEDIAN_BP: float = 1000.0
_DEFAULT_READ_PRIOR_MEDIAN_BP: float = 130.0
# Observation noise std-dev for the fragment length model.
_DEFAULT_SIGMA_OBS: float = 0.05


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    """A sampled path through the minimizer-space de Bruijn graph.

    Attributes:
        minimizer_ids: Ordered minimizer hash IDs along the path.
        distances: Minimizer-count distance between consecutive minimizers
            (always 1 for adjacent minimizers; length = len(minimizer_ids) − 1).
        support: Read-support count for each extension step
            (length = len(minimizer_ids) − 1).
    """

    minimizer_ids: list[int]
    distances: list[int]
    support: list[int]


class _LRUCache:
    """Bounded LRU cache backed by OrderedDict.

    Attributes:
        maxsize: Maximum number of entries before eviction.
        hits: Number of cache hits (for diagnostics).
        misses: Number of cache misses (for diagnostics).
    """

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: object) -> object | None:
        """Return cached value or None on miss, promoting hits to most-recent."""
        try:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        except KeyError:
            self.misses += 1
            return None

    def put(self, key: object, value: object) -> None:
        """Insert or update a cache entry, evicting the oldest if full."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a fraction in [0, 1]."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class FragmentLengthEstimate:
    """Posterior estimate of the log-normal fragment length distribution.

    Attributes:
        mu_log: Posterior mean of the log-scale mean parameter (natural units).
        sigma_log: Posterior mean of the log-scale std-dev parameter.
        mu_log_ci: 95 % credible interval for mu_log.
        sigma_log_ci: 95 % credible interval for sigma_log.
        rho: Posterior mean background contamination fraction.
        norm: Posterior mean normalisation factor (signal amplitude).
        median: Estimated median insert size = exp(mu_log) bp.
        mean: Estimated mean insert size = exp(mu_log + sigma_log²/2) bp.
        n_bins_used: Bins with at least one valid path observation.
        inference: Algorithm used (``"nuts"`` or ``"advi"``).
        signal_reliable: False when containment rates are nearly flat.
    """

    mu_log: float
    sigma_log: float
    mu_log_ci: tuple[float, float]
    sigma_log_ci: tuple[float, float]
    rho: float
    norm: float
    median: float
    mean: float
    n_bins_used: int
    inference: str
    signal_reliable: bool = True
    raw_samples: dict | None = None


@dataclass
class FragmentLengthBPEstimate:
    """Insert-size estimate deconvolved from minimizer-space to basepair space.

    Uses the Poisson-LogNormal variance decomposition: the observed variance
    in minimizer-space is the sum of the true bp-space variance (scaled) and
    Poisson placement noise from random minimizer positions.

    Attributes:
        mu_bp: Posterior mean of log-scale location in bp-space.
        sigma_bp: Posterior mean of log-scale scale in bp-space.
        mu_bp_ci: 95 % credible interval for mu_bp.
        sigma_bp_ci: 95 % credible interval for sigma_bp.
        median_bp: exp(mu_bp) — median insert size in bp.
        mean_bp: exp(mu_bp + sigma_bp²/2) — mean insert size in bp.
        median_bp_ci: 95 % credible interval for median_bp.
        noise_fraction: σ²_noise / σ²_m — fraction of variance from placement noise.
        density: Minimizer density (minimizers per bp) used for conversion.
    """

    mu_bp: float
    sigma_bp: float
    mu_bp_ci: tuple[float, float]
    sigma_bp_ci: tuple[float, float]
    median_bp: float
    mean_bp: float
    median_bp_ci: tuple[float, float]
    noise_fraction: float
    density: float


@dataclass
class FragmentLengthMAP:
    """MAP point estimate of the log-normal fragment length distribution.

    Attributes:
        mu_log: MAP log-scale mean parameter.
        sigma_log: MAP log-scale std-dev parameter.
        rho: MAP background contamination fraction.
        norm: MAP normalisation factor (signal amplitude).
        median: Estimated median insert size = exp(mu_log) bp.
        mean: Estimated mean insert size = exp(mu_log + sigma_log²/2) bp.
        n_bins_used: Bins with at least one valid path observation.
        signal_reliable: False when containment rates are nearly flat.
        loss_final: Final SVI ELBO loss after optimisation.
    """

    mu_log: float
    sigma_log: float
    rho: float
    norm: float
    median: float
    mean: float
    n_bins_used: int
    signal_reliable: bool = True
    loss_final: float = float("nan")


@dataclass
class ReadLengthMAP:
    """MAP point estimate of the log-normal read length distribution.

    Estimated from the bp span between the first and last minimizer within each
    sampled read — a slight underestimate of true read length, but sufficient
    for parameterising downstream models.

    Attributes:
        mu_log: MAP value of the log-scale mean (median read length = exp(mu_log) bp).
        sigma_log: MAP log-scale std-dev.
        median: Estimated median read length = exp(mu_log) bp.
        mean: Estimated mean read length = exp(mu_log + sigma_log²/2) bp.
        n_reads: Number of reads used for estimation.
        signal_reliable: False when fewer than 10 reads had usable spans.
        loss_final: Final SVI ELBO loss after optimisation.
    """

    mu_log: float
    sigma_log: float
    median: float
    mean: float
    n_reads: int
    signal_reliable: bool = True
    loss_final: float = float("nan")


@dataclass
class _BloomFilter:
    """Bloom filter backed by an mmap view of the on-disk bit array.

    The bit array is never fully read into RAM; the OS pages in only the
    ~4 KB pages that are actually accessed.  Repeated probes to the same
    region are served from the OS page cache with no I/O.

    Attributes:
        bits: Numpy uint64 view into the mmap — zero-copy, demand-paged.
        n_bits: Total number of bits in the filter.
        n_hash_fns: Number of hash positions checked per query.
        k: Assembly k stored in the header.
        l: L-mer length stored in the header.
    """

    bits: np.ndarray  # uint64 view into _mm — demand-paged, no copy
    n_bits: int
    n_hash_fns: int
    k: int
    lmer_len: int
    # Backing resources kept alive for the lifetime of this object.
    _mm: mmap.mmap = field(default=None, repr=False)   # type: ignore[assignment]
    _file: IO[bytes] = field(default=None, repr=False)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------

def _open_bloom(path: Path) -> _BloomFilter:
    """Open a minimizer bloom filter with mmap-backed disk access.

    The bit array is memory-mapped; no data is read eagerly.  The OS will
    page in only the ~4 KB pages that cover the three hash positions probed
    per query, keeping RAM usage negligible regardless of the filter size.

    Args:
        path: Path to the ``.minimizer_bloom.bin`` file.

    Returns:
        Bloom filter with a demand-paged numpy view of the bit array.

    Raises:
        ValueError: If the file header is not a valid minimizer bloom filter.
    """
    with open(path, "rb") as f:
        header = f.read(48)
    magic = header[:8]
    if magic == b"MNBLOOMB":
        # 40-byte header: magic(8) + version(8) + l(8) + n_bits(8) + n_hash_fns(8) — no k field.
        _, lmer_len, n_bits, n_hash_fns = struct.unpack_from("<QQQQ", header, offset=8)
        k = 0
        data_offset = 40
    elif magic == b"KMBLOOMB":
        # 48-byte header: magic(8) + version(8) + k(8) + l(8) + n_bits(8) + n_hash_fns(8).
        _, k, lmer_len, n_bits, n_hash_fns = struct.unpack_from("<QQQQQ", header, offset=8)
        data_offset = 48
    else:
        raise ValueError(f"Unknown bloom filter magic: {magic!r}")
    # Keep file and mmap alive via the dataclass fields.
    file_obj: IO[bytes] = open(path, "rb")  # noqa: SIM115
    mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
    # np.frombuffer on an mmap creates a zero-copy view; bits are demand-paged.
    bits = np.frombuffer(mm, dtype=np.uint64, offset=data_offset)
    log.debug("Bloom: n_bits=%d k=%d lmer_len=%d (mmap-backed)", n_bits, k, lmer_len)
    return _BloomFilter(bits=bits, n_bits=n_bits, n_hash_fns=n_hash_fns,
                        k=k, lmer_len=lmer_len, _mm=mm, _file=file_obj)


def _fnv1a64(data: bytes) -> int:
    """Return the FNV-1a 64-bit hash of data, matching the Rust implementation."""
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _bloom_check(bloom: _BloomFilter, kmer_bytes: bytes) -> bool:
    """Return True if kmer_bytes is possibly present in the bloom filter.

    Uses the same FNV-1a double-hashing scheme as the Rust writer.

    Args:
        bloom: Loaded bloom filter.
        kmer_bytes: Canonical k-mer encoded as k × 8 little-endian bytes.

    Returns:
        False if definitely absent; True if possibly present.
    """
    h1 = _fnv1a64(kmer_bytes)
    h2 = (((h1 << 17) | (h1 >> 47)) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    for i in range(3):
        pos = ((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % bloom.n_bits
        if not (int(bloom.bits[pos >> 6]) & (1 << (pos & 63))):
            return False
    return True


def _minimizer_bloom_check(bloom: _BloomFilter, minimizer_hash: int) -> bool:
    """Return True if a single minimizer hash is possibly present in the bloom."""
    return _bloom_check(bloom, struct.pack("<Q", minimizer_hash))


# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------

def _load_shard_paths(prefix: Path, shard_list_suffix: str, single_suffix: str) -> list[Path]:
    """Return shard paths from a shard-list file, or a single-path list if no sharding.

    When rust-mdbg sharding is active it writes a plain-text shard-list file
    (one absolute path per line).  When sharding is inactive that file is absent
    and the single monolithic LMDB path is used instead.

    Args:
        prefix: rust-mdbg output prefix.
        shard_list_suffix: File-name suffix for the shard-list file
            (e.g. ``".index_shard_list"``).
        single_suffix: File-name suffix for the monolithic LMDB
            (e.g. ``".index.lmdb"``).

    Returns:
        Ordered list of LMDB shard paths to open.
    """
    shard_list = Path(f"{prefix}{shard_list_suffix}")
    if shard_list.exists():
        return [Path(p) for p in shard_list.read_text().splitlines() if p.strip()]
    return [Path(f"{prefix}{single_suffix}")]


def _reads_shard_ranges(
    shards: list[tuple],
    read_id_width: int = 4,
) -> list[tuple[int | None, int | None]]:
    """Return the inclusive (lo, hi) read-ID range stored in each reads shard.

    Reads are assigned sequential IDs and shards are filled in order, so each
    shard owns a contiguous ID range.  Ranges are computed from entry counts
    rather than cursor positions: cursor.first()/last() return the
    lexicographically first/last keys, which do NOT correspond to the
    numerically smallest/largest IDs for little-endian integer keys under
    LMDB's default lexicographic sort.

    Args:
        shards: List of ``(env, db, meta_db)`` tuples from
            :func:`_open_reads_lmdb`.
        read_id_width: Unused; kept for API compatibility.

    Returns:
        List of ``(lo, hi)`` pairs (inclusive).  Both values are ``None`` for
        an empty shard.
    """
    ranges: list[tuple[int | None, int | None]] = []
    offset = 0
    for env, db, _ in shards:
        with env.begin() as txn:
            n = txn.stat(db=db)["entries"]
        if n == 0:
            ranges.append((None, None))
        else:
            ranges.append((offset + 1, offset + n))
        offset += n
    return ranges


def _detect_read_id_width(env: "lmdb.Environment", meta_db) -> int:
    """Detect the read ID byte width from LMDB metadata.

    Returns 4 (u32) if ``read_id_width`` metadata is present and equals 4,
    otherwise 8 (u64) for backwards compatibility with older indexes.
    """
    if meta_db is None:
        return 8
    with env.begin(db=meta_db) as txn:
        val = txn.get(b"read_id_width")
        if val is not None and len(val) == 4:
            width = struct.unpack_from("<I", val)[0]
            if width in (4, 8):
                return width
    return 8  # default: legacy u64 format


def _open_reads_lmdb(prefix: Path) -> tuple[list[tuple], int]:
    """Open reads LMDB(s) produced by rust-mdbg --dump-read-minimizers.

    Detects sharding automatically: if ``{prefix}.index_shard_list`` exists,
    opens each shard listed there; otherwise opens the single monolithic LMDB.

    Returns:
        Tuple of (shards, read_id_width) where shards is a list of
        (env, reads_db, meta_db_or_None) tuples — one per shard — and
        read_id_width is 4 (u32) or 8 (u64).
    """
    paths = _load_shard_paths(prefix, ".index_shard_list", ".index.lmdb")
    shards: list[tuple] = []
    read_id_width = 8  # default
    for path in paths:
        env = lmdb.open(
            str(path), readonly=True, lock=False,
            max_dbs=_LMDB_MAX_DBS, map_size=_LMDB_MAP_SIZE,
        )
        db = env.open_db(b"reads")
        try:
            meta_db = env.open_db(b"meta")
        except lmdb.ReadonlyError:
            meta_db = None
        if meta_db is not None and read_id_width == 8:
            read_id_width = _detect_read_id_width(env, meta_db)
        shards.append((env, db, meta_db))
    return shards, read_id_width


def _open_minimizer_lmdb(prefix: Path) -> tuple[list[tuple], int]:
    """Open minimizer index LMDB(s) produced by rust-mdbg --dump-minimizer-index.

    The ``minimizers`` sub-database uses DUPSORT layout:
      key   = minimizer hash (u64 LE, 8 bytes)
      value = 1-based read index (u32 or u64 LE)

    Detects sharding automatically via ``{prefix}.minimizer_index_shard_list``.

    Returns:
        Tuple of (shards, read_id_width) where shards is a list of
        (env, minimizers_db, meta_db_or_None) tuples — one per shard — and
        read_id_width is 4 (u32) or 8 (u64).
    """
    paths = _load_shard_paths(
        prefix, ".minimizer_index_shard_list", ".minimizer_index.lmdb",
    )
    shards: list[tuple] = []
    read_id_width = 8  # default
    for path in paths:
        env = lmdb.open(
            str(path), readonly=True, lock=False,
            max_dbs=_LMDB_MAX_DBS, map_size=_LMDB_MAP_SIZE,
        )
        db = env.open_db(b"minimizers", dupsort=True, dupfixed=True)
        try:
            meta_db = env.open_db(b"meta")
        except lmdb.ReadonlyError:
            meta_db = None
        if meta_db is not None and read_id_width == 8:
            read_id_width = _detect_read_id_width(env, meta_db)
        shards.append((env, db, meta_db))
    return shards, read_id_width


def _decode_read(val: bytes) -> np.ndarray:
    """Decode a reads LMDB value into minimizer_ids (uint64 array).

    Args:
        val: Raw LMDB value bytes — packed u64 LE hashes, 8 bytes each.

    Returns:
        Array of minimizer hash IDs (uint64).
    """
    n = len(val) // 8
    if n == 0:
        return np.empty(0, np.uint64)
    return np.frombuffer(val[: n * 8], dtype="<u8").copy()


def _read_key(read_id: int, width: int = 4) -> bytes:
    """Encode a read ID as an LMDB key (u32 or u64 LE, matching rust-mdbg).

    Args:
        read_id: 1-based read index.
        width: Byte width — 4 (u32, default) or 8 (u64, legacy).
    """
    if width == 4:
        return struct.pack("<I", read_id)
    return struct.pack("<Q", read_id)


def _get_read(
    txn: lmdb.Transaction,
    reads_db: object,
    read_id: int,
    read_id_width: int = 4,
) -> bytes | None:
    """Fetch a read value, trying LE-bytes key first then string key for compat."""
    val = txn.get(_read_key(read_id, read_id_width), db=reads_db)
    if val is None:
        # Try the other width for cross-version compat.
        alt_width = 8 if read_id_width == 4 else 4
        val = txn.get(_read_key(read_id, alt_width), db=reads_db)
    if val is None:
        val = txn.get(str(read_id).encode(), db=reads_db)
    return val


def _n_reads_from_shards(
    reads_txns_dbs_metas: list[tuple],
) -> int | None:
    """Return the global read count from the first shard that has meta.

    Args:
        reads_txns_dbs_metas: List of ``(txn, db, meta_db_or_None)`` tuples.

    Returns:
        Total read count, or ``None`` if no meta sub-database was found.
    """
    for txn, _db, meta_db in reads_txns_dbs_metas:
        if meta_db is None:
            continue
        val = txn.get(b"n_reads", db=meta_db)
        if val is not None:
            return struct.unpack_from("<Q", val)[0]
    return None


def _sample_read_ids(
    reads_txns_dbs_metas: list[tuple],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Return a list of up to n random 1-based read IDs.

    Prefers the fast path (sample from 1..n_reads using metadata) when
    available; falls back to scanning all LMDB keys across all shards.

    Args:
        reads_txns_dbs_metas: List of ``(txn, db, meta_db_or_None)`` tuples —
            one per reads shard.
        n: Number of IDs to sample.
        rng: Random number generator.

    Returns:
        List of sampled read IDs (may contain IDs that map to no entry).
    """
    total = _n_reads_from_shards(reads_txns_dbs_metas)
    if total is not None and total > 0:
        return [rng.randint(1, total) for _ in range(min(n, total))]
    # Fallback: scan all keys across all shards (slow for large databases).
    keys: list[int] = []
    for txn, db, _meta in reads_txns_dbs_metas:
        cursor = txn.cursor(db=db)
        for key, _ in cursor.iternext():
            if len(key) == 4:
                keys.append(struct.unpack("<I", key)[0])
            elif len(key) == 8:
                keys.append(struct.unpack("<Q", key)[0])
            else:
                try:
                    keys.append(int(key))
                except (ValueError, UnicodeDecodeError):
                    continue
    return rng.sample(keys, min(n, len(keys)))


def _read_ids_for_minimizer_multi(
    mi_txns_dbs: list[tuple],
    minimizer_hash: int,
    limit: int,
    read_id_width: int = 4,
) -> list[int]:
    """Return up to *limit* 1-based read IDs containing *minimizer_hash*.

    Queries all minimizer index shards and concatenates results.

    Args:
        mi_txns_dbs: List of ``(txn, db)`` tuples — one per minimizer shard.
        minimizer_hash: Single minimizer hash (u64).
        limit: Maximum number of read IDs to return (across all shards).
        read_id_width: Byte width of read ID values (4 or 8).

    Returns:
        List of read IDs; empty if the minimizer is not in any shard.
    """
    key = struct.pack("<Q", minimizer_hash)
    val_fmt = "<I" if read_id_width == 4 else "<Q"
    ids: list[int] = []
    for txn, db in mi_txns_dbs:
        cursor = txn.cursor(db=db)
        if not cursor.set_key(key):
            continue
        for val in cursor.iternext_dup(keys=False):
            ids.append(struct.unpack_from(val_fmt, val)[0])
            if len(ids) >= limit:
                return ids
    return ids


# ---------------------------------------------------------------------------
# Extension logic
# ---------------------------------------------------------------------------

def _get_read_multi(
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    read_id: int,
    read_id_width: int = 4,
) -> bytes | None:
    """Fetch a raw read value across sharded reads LMDB transactions.

    Uses precomputed per-shard key ranges to route the lookup to the correct
    shard in O(n_shards) time; falls back to trying all shards when ranges are
    unavailable.

    Args:
        reads_txns_dbs: List of ``(txn, db)`` pairs — one per reads shard.
        shard_ranges: Precomputed ``(lo, hi)`` inclusive read-ID range per shard.
        read_id: 1-based read ID to look up.
        read_id_width: Byte width of read ID keys (4 or 8).

    Returns:
        Raw LMDB value bytes, or ``None`` if not found.
    """
    key = _read_key(read_id, read_id_width)
    for (lo, hi), (txn, db) in zip(shard_ranges, reads_txns_dbs):
        if lo is not None and not (lo <= read_id <= hi):
            continue
        val = txn.get(key, db=db)
        if val is not None:
            return val
        if lo is not None:
            # This is the right shard but the key is absent.
            break
        # String-key fallback for legacy non-sharded indexes.
        val = txn.get(str(read_id).encode(), db=db)
        if val is not None:
            return val
    return None


def _get_read_cached(
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    read_id: int,
    cache: _LRUCache,
    read_id_width: int = 4,
) -> np.ndarray | None:
    """Fetch and decode a read across sharded reads LMDB transactions.

    Uses *cache* to avoid repeated LMDB lookups for the same read ID.
    """
    cached = cache.get(read_id)
    if cached is not None:
        return cached  # type: ignore[return-value]
    val = _get_read_multi(reads_txns_dbs, shard_ranges, read_id, read_id_width)
    if val is None:
        return None
    result = _decode_read(val)
    cache.put(read_id, result)
    return result


def _get_minimizer_reads_cached(
    mi_txns_dbs: list[tuple],
    minimizer_hash: int,
    limit: int,
    cache: _LRUCache,
    read_id_width: int = 4,
) -> list[int]:
    """Look up reads containing *minimizer_hash* across all minimizer shards.

    Uses *cache* to avoid repeated LMDB lookups for the same hash.
    """
    cached = cache.get(minimizer_hash)
    if cached is not None:
        return cached  # type: ignore[return-value]
    result = _read_ids_for_minimizer_multi(mi_txns_dbs, minimizer_hash, limit, read_id_width)
    cache.put(minimizer_hash, result)
    return result


def _find_extensions(
    tip: np.ndarray,
    tip_rev: np.ndarray,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    k: int,
    successor: bool,
    max_reads: int,
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    bloom: _BloomFilter | None,
    n_filter_mers: int | None = None,
    read_id_width: int = 4,
) -> tuple[Counter, dict[int, list[int]]]:
    """Find candidate extensions of tip via the minimizer index.

    Selects ``n_filter_mers`` minimizers from the overlap region of the tip,
    fetches the set of read IDs for each from the minimizer index, then
    intersects those sets.  Using more minimizers produces a smaller candidate
    set before the full (k-1) overlap verification, reducing spurious work.

    Intermediate intersection results are cached by their suffix tuple so that
    consecutive extension steps (which share k-2 of the k-1 filter minimizers)
    each require only one new set intersection rather than k-2.

    Args:
        tip: Current k-mer oriented in path direction (uint64 array, length k).
        tip_rev: tip[::-1], pre-computed.
        reads_txns_dbs: Sharded reads LMDB transaction/db pairs.
        shard_ranges: Per-shard key ranges for routing lookups.
        mi_txns_dbs: Sharded minimizer index transaction/db pairs.
        k: De Bruijn graph order.
        successor: If True, find what comes after tip (forward extension);
            if False, find what comes before tip (backward extension).
        max_reads: Maximum reads to examine per lookup.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for suffix-tuple → frozenset(read IDs).
        bloom: Minimizer bloom filter, or None.
        n_filter_mers: Number of overlap minimizers to intersect before
            verifying the full overlap.  ``None`` (default) uses k-1
            (the complete overlap).  Clamped to [1, k-1].

    Returns:
        Tuple of (extension_counts, extension_distances) where
        extension_distances maps each candidate minimizer hash to the list
        of minimizer-count distances (always 1) across reads.
    """
    # Determine how many overlap minimizers to use for read-set intersection.
    n_mers = (k - 1) if n_filter_mers is None else max(1, min(n_filter_mers, k - 1))

    # Select filter minimizers from the overlap region.
    # Forward: overlap is tip[1:]; pick the last n_mers → tip[k-n_mers:]
    # Backward: overlap is tip[:-1]; pick the first n_mers → tip[:n_mers]
    if successor:
        filter_mers = [int(m) for m in tip[k - n_mers:]]
    else:
        filter_mers = [int(m) for m in tip[:n_mers]]

    # Bloom pre-filter each candidate minimizer.
    if bloom is not None:
        filter_mers = [m for m in filter_mers if _minimizer_bloom_check(bloom, m)]
    if not filter_mers:
        return Counter(), {}

    # Build the intersection progressively from filter_mers[-1] (nearest the
    # extension point) inward, caching each suffix tuple.  This lets the next
    # extension step (whose filter drops filter_mers[0] and appends one new
    # minimizer) reuse the cached suffix and perform only one new intersection.
    read_ids: set[int] | None = None
    build_from: int = len(filter_mers) - 2  # index to start extending leftward from

    # Find the longest cached suffix of filter_mers.
    for i in range(len(filter_mers)):
        suffix_key = tuple(filter_mers[i:])
        cached = intersection_cache.get(suffix_key)
        if cached is not None:
            read_ids = set(cached)
            build_from = i - 1
            break

    if read_ids is None:
        # No cache hit: seed from the rightmost (outermost) minimizer.
        first_ids = _get_minimizer_reads_cached(
            mi_txns_dbs, filter_mers[-1], max_reads, minimizer_cache, read_id_width,
        )
        if not first_ids:
            return Counter(), {}
        read_ids = set(first_ids)
        intersection_cache.put((filter_mers[-1],), frozenset(read_ids))
        build_from = len(filter_mers) - 2

    # Extend leftward, caching each new suffix intersection.
    for i in range(build_from, -1, -1):
        m_ids = _get_minimizer_reads_cached(
            mi_txns_dbs, filter_mers[i], max_reads, minimizer_cache, read_id_width,
        )
        if not m_ids:
            return Counter(), {}
        read_ids &= set(m_ids)
        if not read_ids:
            return Counter(), {}
        intersection_cache.put(tuple(filter_mers[i:]), frozenset(read_ids))

    counts: Counter = Counter()
    dists: dict[int, list[int]] = defaultdict(list)

    for read_id in read_ids:
        mids = _get_read_cached(reads_txns_dbs, shard_ranges, read_id, read_cache, read_id_width)
        if mids is None:
            continue
        n = len(mids)
        if n <= k:
            continue

        for i in range(n - k):
            window = mids[i : i + k]
            if np.array_equal(window, tip):
                # Read direction matches path direction.
                if successor and i + k < n:
                    ext = int(mids[i + k])
                    counts[ext] += 1
                    dists[ext].append(1)
                elif not successor and i > 0:
                    ext = int(mids[i - 1])
                    counts[ext] += 1
                    dists[ext].append(1)
            elif np.array_equal(window, tip_rev):
                # Read direction is opposite to path direction.
                if successor and i > 0:
                    ext = int(mids[i - 1])
                    counts[ext] += 1
                    dists[ext].append(1)
                elif not successor and i + k < n:
                    ext = int(mids[i + k])
                    counts[ext] += 1
                    dists[ext].append(1)

    return counts, dists


def _extend_path(
    path: list[int],
    distances: list[int],
    support: list[int],
    direction: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    bloom: _BloomFilter | None,
    k: int,
    max_path_mers: int,
    min_support: int,
    max_reads_per_kmer: int,
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    n_filter_mers: int | None = None,
    read_id_width: int = 4,
) -> None:
    """Extend path in-place in the given direction until a dead end or limit.

    At each step the function looks up candidate next minimizers via the
    minimizer index with (k-1) overlap verification, then accepts the
    extension only when a single unambiguous candidate exceeds min_support.

    Args:
        path: Minimizer ID sequence, modified in place.
        distances: Minimizer-count distances between consecutive minimizers,
            modified in place.
        support: Read-support counts per extension, modified in place.
        direction: +1 to extend at the path tail; -1 to extend at the path head.
        reads_txns_dbs: Sharded reads LMDB transaction/db pairs.
        shard_ranges: Per-shard key ranges for routing lookups.
        mi_txns_dbs: Sharded minimizer index transaction/db pairs.
        bloom: Minimizer bloom filter for pre-filtering, or None.
        k: De Bruijn graph order.
        max_path_mers: Hard cap on path length (minimizer count).
        min_support: Minimum read-support count to accept an extension.
        max_reads_per_kmer: Maximum reads to examine per minimizer lookup.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for suffix-tuple → frozenset(read IDs).
        n_filter_mers: Overlap minimizers to intersect; None = k-1 (full overlap).
    """
    successor = direction > 0

    while len(path) < max_path_mers:
        # The tip is the terminal k-mer of the path in path direction.
        tip_list = path[-k:] if direction > 0 else path[:k][::-1]
        tip = np.array(tip_list, dtype=np.uint64)
        tip_rev = tip[::-1]

        counts, dists = _find_extensions(
            tip, tip_rev,
            reads_txns_dbs, shard_ranges,
            mi_txns_dbs, k, successor, max_reads_per_kmer,
            read_cache, minimizer_cache, intersection_cache, bloom, n_filter_mers,
            read_id_width,
        )
        if not counts:
            break

        # Accept only an unambiguous extension above min_support.
        top2 = counts.most_common(2)
        best_mid, best_count = top2[0]
        if best_count < min_support:
            break
        if len(top2) > 1 and top2[1][1] >= best_count:
            break  # ambiguous branch

        # Validate the new edge minimizer exists in the index.
        if bloom is not None and not _minimizer_bloom_check(bloom, best_mid):
            break

        median_dist = int(np.median(dists[best_mid])) if dists[best_mid] else 0

        if direction > 0:
            path.append(best_mid)
            distances.append(median_dist)
            support.append(best_count)
        else:
            path.insert(0, best_mid)
            distances.insert(0, median_dist)
            support.insert(0, best_count)


def _sample_path(
    seed_read_id: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    bloom: _BloomFilter | None,
    k: int,
    max_path_mers: int,
    min_support: int,
    max_reads_per_kmer: int,
    rng: random.Random,
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    n_filter_mers: int | None = None,
    read_id_width: int = 4,
) -> PathResult | None:
    """Sample one path seeded from a random k-mer window in a given read.

    Initialises the path from a random k-minimizer window in the seed read,
    then extends it both forward and backward until dead ends or the length
    cap is reached.

    Args:
        seed_read_id: 1-based read index to use as the path seed.
        reads_txns_dbs: Sharded reads LMDB transaction/db pairs.
        shard_ranges: Per-shard key ranges for routing lookups.
        mi_txns_dbs: Sharded minimizer index transaction/db pairs.
        bloom: Minimizer bloom filter, or None.
        k: De Bruijn graph order.
        max_path_mers: Maximum path length in minimizers.
        min_support: Minimum read-support for extensions.
        max_reads_per_kmer: Maximum reads to examine per minimizer lookup.
        rng: Random number generator.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for suffix-tuple → frozenset(read IDs).
        n_filter_mers: Overlap minimizers to intersect; None = k-1 (full overlap).

    Returns:
        PathResult, or None if the seed read is shorter than k minimizers.
    """
    mids = _get_read_cached(reads_txns_dbs, shard_ranges, seed_read_id, read_cache, read_id_width)
    if mids is None:
        return None
    if len(mids) < k:
        return None

    start = rng.randrange(len(mids) - k + 1)
    path = list(map(int, mids[start : start + k]))
    distances = [1] * (k - 1)
    support = [1] * (k - 1)

    for direction in (+1, -1):
        _extend_path(
            path, distances, support, direction=direction,
            reads_txns_dbs=reads_txns_dbs, shard_ranges=shard_ranges,
            mi_txns_dbs=mi_txns_dbs, bloom=bloom,
            k=k, max_path_mers=max_path_mers,
            min_support=min_support, max_reads_per_kmer=max_reads_per_kmer,
            read_cache=read_cache, minimizer_cache=minimizer_cache,
            intersection_cache=intersection_cache,
            n_filter_mers=n_filter_mers,
            read_id_width=read_id_width,
        )

    return PathResult(minimizer_ids=path, distances=distances, support=support)


# ---------------------------------------------------------------------------
# Insert size estimation
# ---------------------------------------------------------------------------

def _canonical_combo_hash_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the canonical (commutative) combination hash for each element pair.

    Matches rust-mdbg's ``canonical_combo_hash`` exactly, using the Rust mixing
    constants so that hashes computed here are found in the PE combo LMDB.

    Args:
        a: uint64 array of first minimizer hashes.
        b: uint64 array of second minimizer hashes (same length as *a*).

    Returns:
        uint64 array of canonical combination hashes.
    """
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    h: np.ndarray = lo ^ (hi * _COMBO_CM1)
    h = (h ^ (h >> np.uint64(30))) * _COMBO_CM2
    h = (h ^ (h >> np.uint64(27))) * _COMBO_CM3
    return h ^ (h >> np.uint64(31))


def _open_pe_combo_lmdb(prefix: Path) -> list[tuple]:
    """Open the PE combo LMDB(s) written by rust-mdbg --dump-combo-index (read-only).

    Detects sharding automatically via ``{prefix}.pe_combo_shard_list``.

    Args:
        prefix: rust-mdbg output prefix.

    Returns:
        List of ``(env, db)`` tuples — one per shard.

    Raises:
        FileNotFoundError: If the first (or only) LMDB directory does not exist.
    """
    paths = _load_shard_paths(prefix, ".pe_combo_shard_list", ".pe_combo.lmdb")
    if not paths[0].exists():
        raise FileNotFoundError(f"PE combo LMDB not found: {paths[0]}")
    shards: list[tuple] = []
    for path in paths:
        env = lmdb.open(
            str(path), readonly=True, lock=False,
            max_dbs=2, map_size=_LMDB_MAP_SIZE,
        )
        db = env.open_db(b"combo")
        shards.append((env, db))
    return shards


def _build_path_bin_sketches(
    results: list[PathResult],
    bin_distances: list[float],
    combo_density: float,
) -> list[list[dict[int, int]]]:
    """Build per-path per-bin combination minimizer sketches from sampled paths.

    For every pair of minimizers (i, j) in a path whose cumulative bp distance
    falls within ``[bin_distances[0], bin_distances[-1]]``, computes a canonical
    combination hash using the same function as rust-mdbg's PE combo builder.
    Only hashes ≤ ``combo_density × 2⁶⁴`` are retained.  Hashes that fall into
    more than one bin are discarded (multi-bin conflict).

    Uses exact inter-minimizer bp distances from ``PathResult.distances``; no
    bp-scale approximation is needed.

    Args:
        results: Sampled paths from :func:`_sample_path`.
        bin_distances: Sorted bp breakpoints; length = n_bins + 1.
        combo_density: Thinning fraction — must match the value used to build
            the PE combo LMDB (``--pe-combo-density`` in rust-mdbg).

    Returns:
        ``path_bin_sketches[p][b]`` — dict mapping combo hash → count for path
        *p* and bin *b*.
    """
    if len(bin_distances) < 2:
        raise ValueError("bin_distances must have at least 2 values")

    n_bins = len(bin_distances) - 1
    bin_arr = np.array(bin_distances, dtype=np.float64)
    threshold = np.uint64(int(combo_density * _UINT64_MAX))

    path_bin_sketches: list[list[dict[int, int]]] = []

    for result in results:
        mids = np.array(result.minimizer_ids, dtype=np.uint64)
        n = len(mids)
        if n < 2:
            path_bin_sketches.append([{} for _ in range(n_bins)])
            continue

        # Cumulative bp positions from the inter-minimizer distances.
        positions = np.zeros(n, dtype=np.int64)
        if result.distances:
            positions[1:] = np.cumsum(result.distances, dtype=np.int64)

        # hash → bin index; -1 signals a multi-bin conflict (discard).
        hash_to_bin: dict[int, int] = {}

        for i in range(n):
            pos_i = int(positions[i])
            j_lo = int(np.searchsorted(positions, pos_i + bin_distances[0], side="left"))
            j_hi = int(np.searchsorted(positions, pos_i + bin_distances[-1], side="right")) - 1
            j_lo = max(j_lo, i + 1)
            if j_lo > j_hi:
                continue

            j_range = np.arange(j_lo, j_hi + 1, dtype=np.intp)
            a_arr = np.full(len(j_range), mids[i], dtype=np.uint64)
            hashes = _canonical_combo_hash_v(a_arr, mids[j_range])

            keep = hashes <= threshold
            if not np.any(keep):
                continue

            kept_hashes = hashes[keep]
            kept_dists = (positions[j_range[keep]] - pos_i).astype(np.float64)
            bin_indices = (np.searchsorted(bin_arr, kept_dists, side="right") - 1).tolist()

            for h_val, b_idx in zip(kept_hashes.tolist(), bin_indices):
                h_val = int(h_val)
                if 0 <= b_idx < n_bins:
                    if h_val in hash_to_bin:
                        if hash_to_bin[h_val] != b_idx:
                            hash_to_bin[h_val] = -1  # conflict — will be discarded
                    else:
                        hash_to_bin[h_val] = b_idx

        path_sketch: list[dict[int, int]] = [{} for _ in range(n_bins)]
        for h_val, b_idx in hash_to_bin.items():
            if b_idx >= 0:
                path_sketch[b_idx][h_val] = path_sketch[b_idx].get(h_val, 0) + 1

        path_bin_sketches.append(path_sketch)

    return path_bin_sketches


def _compute_per_path_containment_rates(
    pe_shards: list[tuple],
    path_bin_sketches: list[list[dict[int, int]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-path per-bin containment of path hashes in the PE sketch.

    For each (path, bin) pair queries the PE combo LMDB shards for every hash
    in the path-bin sketch.  Queries are issued individually so the full PE
    sketch is never loaded into RAM.

    A hash is considered present if it is found in **any** shard.  The total
    entry count is the sum of per-shard entry counts (an approximation of the
    unique-hash count, used only as a prior).

    Args:
        pe_shards: List of ``(env, db)`` tuples — one per PE combo shard.
        path_bin_sketches: Per-path per-bin sketch dicts from
            :func:`_build_path_bin_sketches`.

    Returns:
        ``(c_obs, n_hashes, n_pe)`` where

        * ``c_obs``     — float32 ``[n_paths, n_bins]`` containment rates (NaN
          when the bin is empty for that path).
        * ``n_hashes``  — float32 ``[n_paths, n_bins]`` hash counts per path-bin.
        * ``n_pe``      — approximate total unique entries across PE combo shards.
    """
    if not path_bin_sketches:
        return np.empty((0, 0), np.float32), np.empty((0, 0), np.float32), 0

    n_paths = len(path_bin_sketches)
    n_bins = len(path_bin_sketches[0])
    c_obs = np.full((n_paths, n_bins), np.nan, dtype=np.float32)
    n_hashes = np.zeros((n_paths, n_bins), dtype=np.float32)

    with contextlib.ExitStack() as stack:
        pe_txns_dbs = [
            (stack.enter_context(env.begin()), db)  # type: ignore[attr-defined]
            for env, db in pe_shards
        ]
        n_pe: int = sum(txn.stat(db=db)["entries"] for txn, db in pe_txns_dbs)

        for p, sketch_list in enumerate(path_bin_sketches):
            for b, sketch_dict in enumerate(sketch_list):
                n = len(sketch_dict)
                n_hashes[p, b] = n
                if n == 0:
                    continue
                hits = 0
                for h in sketch_dict:
                    key = struct.pack("<Q", h)
                    if any(txn.get(key, db=db) is not None for txn, db in pe_txns_dbs):
                        hits += 1
                c_obs[p, b] = hits / n

    return c_obs, n_hashes, n_pe


@dataclass
class _ContainmentData:
    """Prepared containment data for fragment length inference.

    Attributes:
        c_adjusted: Clipped containment rates ``[n_paths, n_bins]``.
        n_hashes: Per-path per-bin hash counts ``[n_paths, n_bins]``.
        n_pe: Number of unique entries in the PE combo LMDB.
        bin_lo: Lower bin edges (bp) ``[n_bins]``.
        bin_hi: Upper bin edges (bp) ``[n_bins]``.
        observed_mask: Bool ``[n_paths, n_bins]`` where hash count >= threshold.
        n_bins_used: Bins with at least one active path.
        signal_reliable: False when containment is nearly flat.
    """

    c_adjusted: np.ndarray
    n_hashes: np.ndarray
    n_pe: int
    bin_lo: np.ndarray
    bin_hi: np.ndarray
    observed_mask: np.ndarray
    n_bins_used: int
    signal_reliable: bool


def _prepare_containment_data(
    pe_shards: list[tuple],
    path_bin_sketches: list[list[dict[int, int]]],
    bin_distances: list[float],
    min_path_hashes_per_bin: int,
) -> _ContainmentData:
    """Compute containment rates and prepare data for inference.

    Shared by :func:`estimate_fragment_length` and
    :func:`estimate_fragment_length_map`.

    Args:
        pe_shards: List of ``(env, db)`` tuples — one per PE combo shard.
        path_bin_sketches: Per-path per-bin combo hashes.
        bin_distances: Sorted bp breakpoints (length = n_bins + 1).
        min_path_hashes_per_bin: Minimum hashes to include a path-bin.

    Returns:
        Prepared containment data for downstream inference.
    """
    c_obs, n_hashes, n_pe = _compute_per_path_containment_rates(
        pe_shards, path_bin_sketches,
    )
    log.info(
        "PE sketch: %d unique hashes; per-path containment (mean/bin): %s",
        n_pe, np.round(np.nanmean(c_obs, axis=0), 4),
    )
    bin_lo = np.array(bin_distances[:-1], dtype=np.float32)
    bin_hi = np.array(bin_distances[1:], dtype=np.float32)
    c_adjusted = np.clip(
        np.where(np.isnan(c_obs), 0.0, c_obs), 0.0, 1.0,
    ).astype(np.float32)
    observed_mask = n_hashes >= min_path_hashes_per_bin
    n_bins_used = int(observed_mask.any(axis=0).sum())
    signal_reliable = True
    if c_adjusted.size > 1:
        c_range = float(
            (c_adjusted * observed_mask).max()
            - (c_adjusted * observed_mask).min()
        )
        if c_range < 0.10:
            signal_reliable = False
            log.warning(
                "Containment rates nearly flat (range=%.3f); estimate may be prior-dominated.",
                c_range,
            )
    return _ContainmentData(
        c_adjusted=c_adjusted, n_hashes=n_hashes, n_pe=n_pe,
        bin_lo=bin_lo, bin_hi=bin_hi,
        observed_mask=observed_mask, n_bins_used=n_bins_used,
        signal_reliable=signal_reliable,
    )


def _expected_containment_kernel(
    bin_lo: object,
    bin_hi: object,
    mu_log: object,
    sigma_log: object,
    read_length: float,
    combo_max_distance: int | None = None,
    n_quad: int = 200,
) -> object:
    """Compute expected containment per bin using the PE geometry kernel.

    For two minimizers at genomic distance *d* apart, the probability they
    appear on opposite reads (R1/R2) of a PE pair with insert size *I* and
    read length *R* is proportional to the triangle kernel::

        K(d, I, R) = max(0, min(d, R, I-d, I-R))

    When ``combo_max_distance`` is small (<=R), the rust-mdbg filter
    ``|pa - pb| <= D`` collapses the kernel to a rectangular window: K ~ 1
    for ``d ∈ (0, I)``.  In that regime the expected containment in a bin
    simplifies to ``CDF(d_mid + 2R) - CDF(d_mid)``.

    Args:
        bin_lo:  Tensor ``[n_bins]`` lower bin edges (bp).
        bin_hi:  Tensor ``[n_bins]`` upper bin edges (bp).
        mu_log:  Scalar tensor — log-scale mean of insert size.
        sigma_log: Scalar tensor — log-scale std dev.
        read_length: Estimated read length in bp.
        combo_max_distance: The ``--combo-max-distance`` value used in
            rust-mdbg.  If None or > read_length, use the full triangle
            kernel; otherwise use the rectangular approximation.
        n_quad: Number of quadrature points for integration over I.

    Returns:
        Tensor ``[n_bins]`` of unnormalised expected containment per bin.
    """
    R = float(read_length)
    use_triangle = combo_max_distance is None or combo_max_distance > R

    if not use_triangle:
        # Rectangular kernel: containment at distance d ∝ P(d < I < d + 2R).
        d_mid = (bin_lo + bin_hi) / 2.0  # type: ignore[operator]
        lognormal = _torch.distributions.LogNormal(mu_log, sigma_log)  # type: ignore[union-attr]
        return lognormal.cdf(d_mid + 2.0 * R) - lognormal.cdf(  # type: ignore[union-attr]
            _torch.clamp(d_mid, min=1.0),  # type: ignore[union-attr]
        )

    # Full triangle kernel via numerical quadrature over I.
    sigma_clamped = _torch.clamp(sigma_log, min=0.05)  # type: ignore[union-attr]
    lo = _torch.exp(mu_log - 4.0 * sigma_clamped)  # type: ignore[union-attr]
    hi = _torch.exp(mu_log + 4.0 * sigma_clamped)  # type: ignore[union-attr]
    I_grid = _torch.linspace(  # type: ignore[union-attr]
        float(lo.detach()), float(hi.detach()), n_quad,
    )
    lognormal = _torch.distributions.LogNormal(mu_log, sigma_log)  # type: ignore[union-attr]
    log_f = lognormal.log_prob(I_grid)
    f_I = _torch.exp(log_f)  # type: ignore[union-attr]  # [n_quad]
    dI = I_grid[1] - I_grid[0]  # uniform spacing

    # Bin midpoints as representative d values.
    d_mid = (bin_lo + bin_hi) / 2.0  # type: ignore[operator]  # [n_bins]
    d = d_mid.unsqueeze(1)  # [n_bins, 1]
    I_v = I_grid.unsqueeze(0)  # [1, n_quad]
    R_t = _torch.tensor(R)  # type: ignore[union-attr]

    # K(d, I, R) = max(0, min(d, R, I-d, I-R))
    K = _torch.clamp(  # type: ignore[union-attr]
        _torch.minimum(  # type: ignore[union-attr]
            _torch.minimum(d.expand_as(I_v + d), R_t),  # type: ignore[union-attr]
            _torch.minimum(I_v - d, I_v - R_t),  # type: ignore[union-attr]
        ),
        min=0.0,
    )  # [n_bins, n_quad]

    # Integrate over I: sum K * f(I) * dI for each bin.
    return (K * f_I.unsqueeze(0) * dI).sum(dim=1)  # type: ignore[union-attr]  # [n_bins]


def _fragment_length_pyro_model(
    c_adjusted: object,
    bin_lo: object,
    bin_hi: object,
    _n_path_hashes: object,  # pyright: ignore[reportUnusedParameter]
    observed_mask: object,
    read_length: float = 150.0,
    combo_max_distance: int | None = None,
    prior_mu_log: float = math.log(500.0),
) -> None:
    """Pyro generative model for fragment length from per-path containment rates.

    Parameters ``mu_log``, ``sigma_log``, ``rho``, and ``norm`` are shared across
    all paths.  Each (path, bin) pair whose hash count passes the minimum
    threshold contributes one independent Normal observation.

    The likelihood uses a geometry-corrected kernel that accounts for the
    probability that two minimizers *d* apart on the genome end up on
    opposite reads (R1/R2) of a PE pair, rather than the naive CDF-difference
    model.

    Priors::

        mu_log    ~ Normal(log(500), 1.5)  — log-scale mean of insert size
        sigma_log ~ HalfNormal(0.5)
        rho       ~ Beta(1, 20)     — background contamination
        norm      ~ Beta(2, 2)      — signal amplitude

    Args:
        c_adjusted:    Tensor ``[n_paths, n_bins]`` noise-adjusted containment.
        bin_lo:        Tensor ``[n_bins]`` lower bin edges (bp).
        bin_hi:        Tensor ``[n_bins]`` upper bin edges (bp).
        _n_path_hashes: Tensor ``[n_paths, n_bins]`` hash counts (unused here;
            kept for interface consistency).
        observed_mask: Bool tensor ``[n_paths, n_bins]`` active path-bin pairs.
        read_length:   Estimated read length in bp.
        combo_max_distance: The ``--combo-max-distance`` used in rust-mdbg.
    """
    mu_log = _pyro.sample(  # type: ignore[union-attr]
        "mu_log",
        _pyro_dist.Normal(  # type: ignore[union-attr]
            _torch.tensor(prior_mu_log),  # type: ignore[union-attr]
            _torch.tensor(1.5),  # type: ignore[union-attr]
        ),
    )
    sigma_log = _pyro.sample(  # type: ignore[union-attr]
        "sigma_log",
        _pyro_dist.HalfNormal(_torch.tensor(0.5)),  # type: ignore[union-attr]
    )
    rho = _pyro.sample(  # type: ignore[union-attr]
        "rho",
        _pyro_dist.Beta(  # type: ignore[union-attr]
            _torch.tensor(1.0), _torch.tensor(20.0),  # type: ignore[union-attr]
        ),
    )
    norm = _pyro.sample(  # type: ignore[union-attr]
        "norm",
        _pyro_dist.Beta(  # type: ignore[union-attr]
            _torch.tensor(2.0), _torch.tensor(2.0),  # type: ignore[union-attr]
        ),
    )

    p = _expected_containment_kernel(
        bin_lo, bin_hi, mu_log, sigma_log,
        read_length=read_length,
        combo_max_distance=combo_max_distance,
    )  # [n_bins]
    expected = p * norm + rho  # [n_bins], broadcast to [n_paths, n_bins]

    n_paths, n_bins = c_adjusted.shape  # type: ignore[union-attr]
    with _pyro.plate("paths", n_paths, dim=-2):  # type: ignore[union-attr]
        with _pyro.plate("bins", n_bins, dim=-1):  # type: ignore[union-attr]
            with _pyro.poutine.mask(mask=observed_mask):  # type: ignore[union-attr]
                _pyro.sample(  # type: ignore[union-attr]
                    "obs",
                    _pyro_dist.Normal(expected.unsqueeze(0), _DEFAULT_SIGMA_OBS),  # type: ignore[union-attr]
                    obs=c_adjusted,
                )


def estimate_fragment_length(
    pe_shards: list[tuple],
    path_bin_sketches: list[list[dict[int, int]]],
    bin_distances: list[float],
    min_path_hashes_per_bin: int = 50,
    inference: Literal["nuts", "advi"] = "nuts",
    num_samples: int = 500,
    num_warmup: int = 200,
    read_length_bp: float = 150.0,
    combo_max_distance: int | None = None,
    prior_mu_log: float | None = None,
) -> FragmentLengthEstimate:
    """Estimate insert size by fitting a log-normal to per-path containment rates.

    Uses a Bayesian model (NUTS or ADVI) that treats each (path, bin) pair as an
    independent observation weighted by its hash count.  The likelihood uses a
    geometry-corrected kernel that accounts for the probability that two
    minimizers *d* apart on the genome end up on opposite reads of a PE pair.

    Args:
        pe_env: LMDB environment for the PE combo sketch.
        pe_db: ``"combo"`` sub-database handle.
        path_bin_sketches: Per-path per-bin combo hashes from
            :func:`_build_path_bin_sketches`.
        bin_distances: Sorted bp breakpoints (length = n_bins + 1).
        min_path_hashes_per_bin: Minimum hashes in a path-bin to include it.
        inference: Inference backend — ``"nuts"`` (MCMC) or ``"advi"`` (SVI).
        num_samples: Posterior samples to draw.
        num_warmup: Warm-up steps (NUTS) or ignored (ADVI).
        read_length_bp: Estimated read length in bp for the containment
            kernel.
        combo_max_distance: The ``--combo-max-distance`` value used in
            rust-mdbg.  Passed to :func:`_expected_containment_kernel`.

    Returns:
        :class:`FragmentLengthEstimate` with posterior means and credible
        intervals for ``mu_log``, ``sigma_log``, ``rho``, and ``norm``.

    Raises:
        RuntimeError: If pyro-ppl and torch are not installed.
    """
    if not _PYRO_AVAILABLE:
        raise RuntimeError("pyro-ppl and torch are required for insert size estimation.")

    cd = _prepare_containment_data(
        pe_shards, path_bin_sketches, bin_distances, min_path_hashes_per_bin,
    )

    _prior_mu = prior_mu_log if prior_mu_log is not None else math.log(_DEFAULT_FRAG_PRIOR_MEDIAN_BP)
    mu0 = _prior_mu
    if cd.n_bins_used == 0:
        return FragmentLengthEstimate(
            mu_log=mu0, sigma_log=0.5,
            mu_log_ci=(mu0 - 1.0, mu0 + 1.0),
            sigma_log_ci=(0.0, 1.0),
            rho=0.05, norm=0.5,
            median=math.exp(mu0),
            mean=math.exp(mu0 + 0.5**2 / 2.0),
            n_bins_used=0, inference=inference, signal_reliable=False,
        )

    c_adj_t  = _torch.tensor(cd.c_adjusted, dtype=_torch.float32)    # type: ignore[union-attr]
    bin_lo_t = _torch.tensor(cd.bin_lo, dtype=_torch.float32).clamp(min=1.0)  # type: ignore[union-attr]
    bin_hi_t = _torch.tensor(cd.bin_hi, dtype=_torch.float32).clamp(min=1.0)  # type: ignore[union-attr]
    n_path_t = _torch.tensor(cd.n_hashes, dtype=_torch.float32)      # type: ignore[union-attr]
    mask_t   = _torch.tensor(cd.observed_mask, dtype=_torch.bool)     # type: ignore[union-attr]
    model_args = (c_adj_t, bin_lo_t, bin_hi_t, n_path_t, mask_t)
    model_kwargs = {
        "read_length": read_length_bp,
        "combo_max_distance": combo_max_distance,
        "prior_mu_log": _prior_mu,
    }

    init_vals = {
        "mu_log": _torch.tensor(_prior_mu),  # type: ignore[union-attr]
        "sigma_log": _torch.tensor(0.4),                       # type: ignore[union-attr]
        "rho": _torch.tensor(0.05),                            # type: ignore[union-attr]
        "norm": _torch.tensor(0.5),                            # type: ignore[union-attr]
    }

    if inference == "nuts":
        _pyro.clear_param_store()  # type: ignore[union-attr]
        kernel = NUTS(_fragment_length_pyro_model)  # type: ignore[call-arg]
        mcmc = MCMC(  # type: ignore[call-arg]
            kernel, num_samples=num_samples, warmup_steps=num_warmup,
            disable_progbar=False,
            initial_params={k: v for k, v in init_vals.items()},
        )
        mcmc.run(*model_args, **model_kwargs)
        samples = mcmc.get_samples()
        raw_samples = {k: v.detach().clone() for k, v in samples.items()}
        mu_samps    = samples["mu_log"].numpy()
        sig_samps   = samples["sigma_log"].numpy()
        rho_samps   = samples["rho"].numpy()
        norm_samps  = samples["norm"].numpy()

        raw_samples_out = raw_samples

    else:  # advi
        raw_samples_out = None
        _pyro.clear_param_store()  # type: ignore[union-attr]
        guide = AutoNormal(  # type: ignore[call-arg]
            _fragment_length_pyro_model,
            init_loc_fn=_pyro.infer.autoguide.init_to_value(values=init_vals),  # type: ignore[union-attr]
        )
        svi_obj = SVI(  # type: ignore[call-arg]
            _fragment_length_pyro_model, guide,
            _PyroAdam({"lr": 0.01}),  # type: ignore[call-arg]
            loss=Trace_ELBO(),  # type: ignore[call-arg]
        )
        for _ in range(2000):
            svi_obj.step(*model_args, **model_kwargs)
        post = Predictive(  # type: ignore[call-arg]
            _fragment_length_pyro_model, guide=guide,
            num_samples=num_samples,
            return_sites=["mu_log", "sigma_log", "rho", "norm"],
        )(*model_args, **model_kwargs)
        mu_samps   = post["mu_log"].squeeze().numpy()
        sig_samps  = post["sigma_log"].squeeze().numpy()
        rho_samps  = post["rho"].squeeze().numpy()
        norm_samps = post["norm"].squeeze().numpy()

    mu_mean  = float(np.mean(mu_samps))
    sig_mean = float(np.mean(sig_samps))
    return FragmentLengthEstimate(
        mu_log=mu_mean,
        sigma_log=sig_mean,
        mu_log_ci=(float(np.percentile(mu_samps, 2.5)), float(np.percentile(mu_samps, 97.5))),
        sigma_log_ci=(float(np.percentile(sig_samps, 2.5)), float(np.percentile(sig_samps, 97.5))),
        rho=float(np.mean(rho_samps)),
        norm=float(np.mean(norm_samps)),
        median=math.exp(mu_mean),
        mean=math.exp(mu_mean + sig_mean**2 / 2.0),
        n_bins_used=cd.n_bins_used,
        inference=inference,
        signal_reliable=cd.signal_reliable,
        raw_samples=raw_samples_out,
    )


def _deconvolve_to_bp_space(
    nuts_samples: dict,
    density: float,
) -> FragmentLengthBPEstimate:
    """Deconvolve minimizer-space insert-size posterior to bp-space.

    Applies the Poisson-LogNormal variance decomposition to each NUTS
    posterior sample of (mu_log, sigma_log).  The observed minimizer-space
    variance is decomposed into true bp-space variance and Poisson placement
    noise from random minimizer positions along the sequence.

    On the log scale:
        μ_m  = μ_bp + log(d)
        σ²_m = σ²_bp + 1/exp(μ_m)     [Poisson noise ≈ 1/median_count]

    Parameters
    ----------
    nuts_samples : dict
        NUTS posterior samples with keys ``"mu_log"`` and ``"sigma_log"``.
    density : float
        Minimizer density (minimizers per bp).

    Returns
    -------
    FragmentLengthBPEstimate
    """
    mu_m = nuts_samples["mu_log"]        # [n_samples]
    sigma_m = nuts_samples["sigma_log"]  # [n_samples]

    log_d = math.log(density)

    # Location shift: μ_bp = μ_m − log(d)
    mu_bp = mu_m - log_d

    # Poisson placement noise variance on log scale: 1/median_minimizer_count
    sigma_noise_sq = 1.0 / _torch.exp(mu_m)  # type: ignore[union-attr]

    # Variance decomposition: σ²_bp = σ²_m − σ²_noise
    sigma_bp_sq = _torch.clamp(sigma_m ** 2 - sigma_noise_sq, min=0.0)  # type: ignore[union-attr]
    sigma_bp = _torch.sqrt(sigma_bp_sq)  # type: ignore[union-attr]

    # Derived quantities per sample
    median_bp = _torch.exp(mu_bp)  # type: ignore[union-attr]
    mean_bp = _torch.exp(mu_bp + sigma_bp_sq / 2)  # type: ignore[union-attr]
    noise_frac = (sigma_noise_sq / _torch.clamp(sigma_m ** 2, min=1e-12)).mean().item()  # type: ignore[union-attr]

    def _ci(t: object) -> tuple[float, float]:
        q = _torch.quantile(t, _torch.tensor([0.025, 0.975]))  # type: ignore[union-attr]
        return (q[0].item(), q[1].item())

    return FragmentLengthBPEstimate(
        mu_bp=mu_bp.mean().item(),
        sigma_bp=sigma_bp.mean().item(),
        mu_bp_ci=_ci(mu_bp),
        sigma_bp_ci=_ci(sigma_bp),
        median_bp=median_bp.mean().item(),
        mean_bp=mean_bp.mean().item(),
        median_bp_ci=_ci(median_bp),
        noise_fraction=noise_frac,
        density=density,
    )


def estimate_fragment_length_map(
    pe_shards: list[tuple],
    path_bin_sketches: list[list[dict[int, int]]],
    bin_distances: list[float],
    min_path_hashes_per_bin: int = 50,
    num_steps: int = 2000,
    lr: float = 0.01,
    read_length_bp: float = 150.0,
    combo_max_distance: int | None = None,
    prior_mu_log: float | None = None,
) -> FragmentLengthMAP:
    """Find MAP parameter values for the fragment length model via SVI.

    Uses an :class:`AutoDelta` guide (one point per latent site) to maximise
    the joint log-probability.  Faster than :func:`estimate_fragment_length` but
    returns a single point estimate rather than a posterior.

    Args:
        pe_env: LMDB environment for the PE combo sketch.
        pe_db: ``"combo"`` sub-database handle.
        path_bin_sketches: Per-path per-bin combo hashes.
        bin_distances: Sorted bp breakpoints (length = n_bins + 1).
        min_path_hashes_per_bin: Minimum hashes in a path-bin to include it.
        num_steps: Number of SVI optimisation steps.
        lr: Adam learning rate.
        read_length_bp: Estimated read length in bp for the containment
            kernel.
        combo_max_distance: The ``--combo-max-distance`` value used in
            rust-mdbg.  Passed to :func:`_expected_containment_kernel`.

    Returns:
        :class:`FragmentLengthMAP` with MAP values of ``mu_log``, ``sigma_log``,
        ``rho``, and ``norm``.

    Raises:
        RuntimeError: If pyro-ppl and torch are not installed.
    """
    if not _PYRO_AVAILABLE:
        raise RuntimeError("pyro-ppl and torch are required for MAP insert size estimation.")

    cd = _prepare_containment_data(
        pe_shards, path_bin_sketches, bin_distances, min_path_hashes_per_bin,
    )

    _prior_mu = prior_mu_log if prior_mu_log is not None else math.log(_DEFAULT_FRAG_PRIOR_MEDIAN_BP)
    mu0 = _prior_mu
    if cd.n_bins_used == 0:
        return FragmentLengthMAP(
            mu_log=mu0, sigma_log=0.5, rho=0.1, norm=0.5,
            median=math.exp(mu0), mean=math.exp(mu0 + 0.5**2 / 2.0),
            n_bins_used=0, signal_reliable=False,
        )

    c_adj_t  = _torch.tensor(cd.c_adjusted, dtype=_torch.float32)       # type: ignore[union-attr]
    bin_lo_t = _torch.tensor(cd.bin_lo, dtype=_torch.float32).clamp(min=1.0)  # type: ignore[union-attr]
    bin_hi_t = _torch.tensor(cd.bin_hi, dtype=_torch.float32).clamp(min=1.0)  # type: ignore[union-attr]
    n_path_t = _torch.tensor(cd.n_hashes, dtype=_torch.float32)          # type: ignore[union-attr]
    mask_t   = _torch.tensor(cd.observed_mask, dtype=_torch.bool)         # type: ignore[union-attr]
    model_args = (c_adj_t, bin_lo_t, bin_hi_t, n_path_t, mask_t)
    model_kwargs = {
        "read_length": read_length_bp,
        "combo_max_distance": combo_max_distance,
        "prior_mu_log": _prior_mu,
    }

    _pyro.clear_param_store()  # type: ignore[union-attr]
    guide = AutoDelta(_fragment_length_pyro_model)  # type: ignore[call-arg]
    svi = SVI(  # type: ignore[call-arg]
        _fragment_length_pyro_model, guide,
        _PyroAdam({"lr": lr}),  # type: ignore[call-arg]
        loss=Trace_ELBO(),  # type: ignore[call-arg]
    )
    loss_final = float("nan")
    for _ in range(num_steps):
        loss_final = svi.step(*model_args, **model_kwargs)

    map_post = Predictive(  # type: ignore[call-arg]
        _fragment_length_pyro_model, guide=guide,
        num_samples=1, return_sites=["mu_log", "sigma_log", "rho", "norm"],
    )(*model_args, **model_kwargs)

    mu_map  = float(map_post["mu_log"].squeeze())
    sig_map = float(map_post["sigma_log"].squeeze())
    return FragmentLengthMAP(
        mu_log=mu_map,
        sigma_log=sig_map,
        rho=float(map_post["rho"].squeeze()),
        norm=float(map_post["norm"].squeeze()),
        median=math.exp(mu_map),
        mean=math.exp(mu_map + sig_map**2 / 2.0),
        n_bins_used=cd.n_bins_used,
        signal_reliable=cd.signal_reliable,
        loss_final=loss_final,
    )


def _read_length_pyro_model(spans: object) -> None:
    """Pyro generative model for read length from minimizer-span observations.

    Each read's minimizer span (bp distance from first to last minimizer) is
    treated as an independent draw from LogNormal(mu_log, sigma_log), where
    mu_log is the log-scale mean (= log of the median span in bp).

    Priors::

        mu_log    ~ Normal(log(130), 0.5)   — prior centered on log(130 bp)
        sigma_log ~ HalfNormal(0.3)         — reads are near-fixed length

    Args:
        spans: Tensor ``[n_reads]`` of minimizer-span lengths in bp.
    """
    mu_log = _pyro.sample(  # type: ignore[union-attr]
        "mu_log",
        _pyro_dist.Normal(  # type: ignore[union-attr]
            _torch.tensor(np.log(_DEFAULT_READ_PRIOR_MEDIAN_BP)),  # type: ignore[union-attr]
            _torch.tensor(0.5),  # type: ignore[union-attr]
        ),
    )
    sigma_log = _pyro.sample(  # type: ignore[union-attr]
        "sigma_log",
        _pyro_dist.HalfNormal(_torch.tensor(0.3)),  # type: ignore[union-attr]
    )
    with _pyro.plate("reads", spans.shape[0]):  # type: ignore[union-attr]
        _pyro.sample(  # type: ignore[union-attr]
            "obs",
            _pyro_dist.LogNormal(  # type: ignore[union-attr]
                mu_log, sigma_log,  # type: ignore[union-attr]
            ),
            obs=spans,
        )


def estimate_read_length_map(
    reads_shards: list[tuple],
    n_reads: int = 1000,
    min_minimizers: int = 5,
    num_steps: int = 1000,
    lr: float = 0.01,
    rng: random.Random | None = None,
) -> ReadLengthMAP:
    """Find MAP parameters for the read length distribution (minimizer-count).

    Samples up to *n_reads* reads from the reads LMDB, counts the number of
    minimizers per read as a proxy for read length, then fits a log-normal
    distribution via MAP (AutoDelta + SVI).

    Note: distances are in minimizer-count units, not basepairs.  The bp
    conversion will be added when minimizer-to-basepair mapping is implemented.

    Args:
        reads_shards: List of (env, db, meta_db_or_None) shard tuples.
        n_reads: Number of reads to sample for estimation.
        min_minimizers: Skip reads with fewer minimizers than this.
        num_steps: Number of SVI optimisation steps.
        lr: Adam learning rate.
        rng: Random number generator; a fresh one is seeded from OS entropy
            when *None*.

    Returns:
        :class:`ReadLengthMAP` with MAP values of ``mu_log`` and ``sigma_log``,
        plus derived ``median`` and ``mean`` in minimizer-count units.

    Raises:
        RuntimeError: If pyro-ppl and torch are not installed.
    """
    if not _PYRO_AVAILABLE:
        raise RuntimeError("pyro-ppl and torch are required for read length MAP estimation.")

    if rng is None:
        rng = random.Random()

    # Collect all keys across shards, sample, then fetch values.
    spans: list[float] = []
    all_key_shard: list[tuple[bytes, int]] = []  # (key, shard_idx)
    for shard_idx, (env, db, _) in enumerate(reads_shards):
        with env.begin() as txn:  # type: ignore[union-attr]
            cursor = txn.cursor(db=db)
            for key, _ in cursor.iternext():
                all_key_shard.append((key, shard_idx))
    if not all_key_shard:
        return ReadLengthMAP(
            mu_log=math.log(130.0), sigma_log=0.3,
            median=130.0, mean=math.exp(math.log(130.0) + 0.3**2 / 2.0),
            n_reads=0, signal_reliable=False,
        )
    sample_pairs = rng.sample(all_key_shard, min(n_reads, len(all_key_shard)))
    for key, shard_idx in sample_pairs:
        env, db, _ = reads_shards[shard_idx]
        with env.begin() as txn:  # type: ignore[union-attr]
            val = txn.get(key, db=db)
        if val is None:
            continue
        mids = _decode_read(val)
        if len(mids) < min_minimizers:
            continue
        span = float(len(mids))
        if span > 0:
            spans.append(span)

    n_valid = len(spans)
    log.info("Read length estimation: %d reads with usable minimizer spans", n_valid)
    if n_valid < 10:
        mu0 = math.log(130.0)
        return ReadLengthMAP(
            mu_log=mu0, sigma_log=0.3,
            median=math.exp(mu0),
            mean=math.exp(mu0 + 0.3**2 / 2.0),
            n_reads=n_valid, signal_reliable=False,
        )

    spans_t = _torch.tensor(spans, dtype=_torch.float32)  # type: ignore[union-attr]

    _pyro.clear_param_store()  # type: ignore[union-attr]
    guide = AutoDelta(_read_length_pyro_model)  # type: ignore[call-arg]
    svi = SVI(  # type: ignore[call-arg]
        _read_length_pyro_model, guide,
        _PyroAdam({"lr": lr}),  # type: ignore[call-arg]
        loss=Trace_ELBO(),  # type: ignore[call-arg]
    )
    loss_final = float("nan")
    for _ in range(num_steps):
        loss_final = svi.step(spans_t)

    map_post = Predictive(  # type: ignore[call-arg]
        _read_length_pyro_model, guide=guide,
        num_samples=1, return_sites=["mu_log", "sigma_log"],
    )(spans_t)

    mu_map  = float(map_post["mu_log"].squeeze())
    sig_map = float(map_post["sigma_log"].squeeze())
    return ReadLengthMAP(
        mu_log=mu_map,
        sigma_log=sig_map,
        median=math.exp(mu_map),
        mean=math.exp(mu_map + sig_map**2 / 2.0),
        n_reads=n_valid,
        signal_reliable=True,
        loss_final=loss_final,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


@dataclass
class _IndexHandles:
    """Open LMDB environments and auxiliary indexes for path sampling."""

    reads_shards: list[tuple]       # list of (env, db, meta_db_or_None) per shard
    reads_shard_ranges: list[tuple] # list of (lo, hi) inclusive read-ID ranges
    mi_shards: list[tuple]          # list of (env, db, meta_db_or_None) per shard
    bloom: _BloomFilter | None
    read_id_width: int              # 4 (u32) or 8 (u64) — byte width for read IDs


def _open_all_indexes(
    prefix: Path,
    no_bloom: bool,
) -> _IndexHandles:
    """Open reads LMDB(s), minimizer index LMDB(s), and bloom filter."""
    log.info("Opening reads LMDB: %s.index.lmdb (or shards)", prefix)
    reads_shards, reads_id_width = _open_reads_lmdb(prefix)
    log.info("n_shards=%d, read_id_width=%d", len(reads_shards), reads_id_width)

    log.info("Opening minimizer index LMDB: %s.minimizer_index.lmdb (or shards)", prefix)
    mi_shards, mi_id_width = _open_minimizer_lmdb(prefix)
    log.info("minimizer index n_shards=%d, read_id_width=%d", len(mi_shards), mi_id_width)

    # Both indexes must agree on read ID width.
    read_id_width = reads_id_width
    if mi_id_width != reads_id_width:
        log.warning(
            "Read ID width mismatch: reads=%d, minimizer_index=%d; using reads width",
            reads_id_width, mi_id_width,
        )

    reads_shard_ranges = _reads_shard_ranges(reads_shards, read_id_width)

    bloom: _BloomFilter | None = None
    if not no_bloom:
        bloom_path = Path(f"{prefix}.minimizer_bloom.bin")
        if bloom_path.exists():
            log.info("Loading minimizer bloom filter: %s", bloom_path)
            bloom = _open_bloom(bloom_path)
        else:
            log.warning("Bloom filter not found at %s; proceeding without.", bloom_path)

    return _IndexHandles(
        reads_shards=reads_shards,
        reads_shard_ranges=reads_shard_ranges,
        mi_shards=mi_shards,
        bloom=bloom,
        read_id_width=read_id_width,
    )


_READ_CACHE_SIZE = 100_000
_MINIMIZER_CACHE_SIZE = 50_000
_INTERSECTION_CACHE_SIZE = 10_000


def _run_path_sampling(
    idx: _IndexHandles,
    *,
    k: int,
    n_paths: int,
    max_path_mers: int,
    min_support: int,
    max_reads_per_kmer: int,
    rng: random.Random,
    out_fp: IO[str],
    n_filter_mers: int | None = None,
) -> tuple[list[PathResult], int, int]:
    """Sample paths and write JSONL output.

    Returns:
        Tuple of (sampled_results, n_written, attempts).
    """
    n_written = 0
    attempts = 0
    sampled_results: list[PathResult] = []

    read_cache = _LRUCache(maxsize=_READ_CACHE_SIZE)
    minimizer_cache = _LRUCache(maxsize=_MINIMIZER_CACHE_SIZE)
    intersection_cache = _LRUCache(maxsize=_INTERSECTION_CACHE_SIZE)

    with contextlib.ExitStack() as stack:
        reads_txns_dbs = [
            (stack.enter_context(env.begin()), db)
            for env, db, _ in idx.reads_shards
        ]
        reads_txns_dbs_metas = [
            (txn, db, meta_db)
            for (txn, db), (_, _, meta_db) in zip(reads_txns_dbs, idx.reads_shards)
        ]
        mi_txns_dbs = [
            (stack.enter_context(env.begin()), db)
            for env, db, _ in idx.mi_shards
        ]

        seed_ids = _sample_read_ids(
            reads_txns_dbs_metas,
            n=min(n_paths * 10, 100_000),
            rng=rng,
        )
        if not seed_ids:
            log.error("No reads found in reads LMDB")
            raise typer.Exit(1)

        log.info("Sampled %d candidate seed reads; targeting %d paths", len(seed_ids), n_paths)

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Sampling paths", total=n_paths)
            for seed_id in seed_ids:
                if n_written >= n_paths:
                    break
                attempts += 1

                result = _sample_path(
                    seed_read_id=seed_id,
                    reads_txns_dbs=reads_txns_dbs,
                    shard_ranges=idx.reads_shard_ranges,
                    mi_txns_dbs=mi_txns_dbs,
                    bloom=idx.bloom,
                    k=k,
                    max_path_mers=max_path_mers,
                    min_support=min_support,
                    max_reads_per_kmer=max_reads_per_kmer,
                    rng=rng,
                    read_cache=read_cache,
                    minimizer_cache=minimizer_cache,
                    intersection_cache=intersection_cache,
                    n_filter_mers=n_filter_mers,
                    read_id_width=idx.read_id_width,
                )
                if result is None or len(result.minimizer_ids) <= k:
                    continue

                print(
                    json.dumps({
                        "minimizer_ids": result.minimizer_ids,
                        "distances": result.distances,
                        "support": result.support,
                    }),
                    file=out_fp,
                )
                sampled_results.append(result)
                n_written += 1
                progress.advance(task)

    log.info(
        "Cache stats — read_cache: %d entries, %.1f%% hit rate; "
        "minimizer_cache: %d entries, %.1f%% hit rate; "
        "intersection_cache: %d entries, %.1f%% hit rate",
        len(read_cache), read_cache.hit_rate * 100,
        len(minimizer_cache), minimizer_cache.hit_rate * 100,
        len(intersection_cache), intersection_cache.hit_rate * 100,
    )
    return sampled_results, n_written, attempts


def _run_insert_size_estimation(
    prefix: Path,
    sampled_results: list[PathResult],
    *,
    insert_size_bins: str,
    pe_combo_density: float,
    insert_size_inference: str,
    insert_size_paths: int,
    min_path_hashes_per_bin: int,
    insert_size_json: Path | None,
    read_length_bp: float = 150.0,
    combo_max_distance: int | None = None,
    density: float | None = None,
) -> None:
    """Build combo sketches and run insert size inference."""
    # Accept both sharded (shard-list file) and non-sharded PE combo indexes.
    pe_shard_list = Path(f"{prefix}.pe_combo_shard_list")
    pe_single = Path(f"{prefix}.pe_combo.lmdb")
    if not pe_shard_list.exists() and not pe_single.exists():
        log.warning(
            "PE combo LMDB not found at %s; skipping insert size estimation. "
            "Re-run rust-mdbg with --dump-combo-index.",
            pe_single,
        )
        return
    if not _PYRO_AVAILABLE:
        log.warning("pyro-ppl and torch are not installed; skipping insert size estimation.")
        return

    bin_distances = [float(x) for x in insert_size_bins.split(",")]
    use_paths = sampled_results[:insert_size_paths]
    log.info(
        "Building path combo sketches for %d paths (%d bins, pe_density=%.4f)...",
        len(use_paths), len(bin_distances) - 1, pe_combo_density,
    )
    path_bin_sketches = _build_path_bin_sketches(use_paths, bin_distances, pe_combo_density)
    pe_shards = _open_pe_combo_lmdb(prefix)
    log.info("PE combo n_shards=%d", len(pe_shards))

    # Convert read length from bp to minimizer-count units for kernel consistency.
    # All kernel inputs (bin edges, insert size, read length) must share units.
    if density is not None and density > 0:
        read_length_mers = read_length_bp * density
    else:
        read_length_mers = read_length_bp  # fallback: assume units already match
    log.info(
        "Running insert size inference (method=%s, read_length_bp=%.1f, "
        "read_length_mers=%.1f, combo_max_distance=%s, density=%s)...",
        insert_size_inference, read_length_bp, read_length_mers,
        combo_max_distance, density,
    )
    if insert_size_inference == "map":
        result_is = estimate_fragment_length_map(
            pe_shards, path_bin_sketches, bin_distances,
            min_path_hashes_per_bin=min_path_hashes_per_bin,
            read_length_bp=read_length_mers,
            combo_max_distance=combo_max_distance,
        )
        is_dict: dict[str, object] = {
            "method": "map",
            "mu_log": result_is.mu_log,
            "sigma_log": result_is.sigma_log,
            "rho": result_is.rho,
            "norm": result_is.norm,
            "median_bp": result_is.median,
            "mean_bp": result_is.mean,
            "n_bins_used": result_is.n_bins_used,
            "signal_reliable": result_is.signal_reliable,
            "loss_final": result_is.loss_final,
        }
        log.info(
            "Insert size MAP: median=%.0f bp  mean=%.0f bp  "
            "mu_log=%.3f  sigma_log=%.3f  rho=%.4f  norm=%.4f  "
            "bins_used=%d  signal_reliable=%s",
            result_is.median, result_is.mean,
            result_is.mu_log, result_is.sigma_log,
            result_is.rho, result_is.norm,
            result_is.n_bins_used, result_is.signal_reliable,
        )
    else:
        result_is_full = estimate_fragment_length(
            pe_shards, path_bin_sketches, bin_distances,
            min_path_hashes_per_bin=min_path_hashes_per_bin,
            inference=insert_size_inference,  # type: ignore[arg-type]
            read_length_bp=read_length_mers,
            combo_max_distance=combo_max_distance,
        )
        is_dict = {
            "method": result_is_full.inference,
            "mu_log": result_is_full.mu_log,
            "sigma_log": result_is_full.sigma_log,
            "mu_log_ci": list(result_is_full.mu_log_ci),
            "sigma_log_ci": list(result_is_full.sigma_log_ci),
            "rho": result_is_full.rho,
            "norm": result_is_full.norm,
            "median_bp": result_is_full.median,
            "mean_bp": result_is_full.mean,
            "n_bins_used": result_is_full.n_bins_used,
            "signal_reliable": result_is_full.signal_reliable,
        }
        log.info(
            "Insert size %s: median=%.0f bp  mean=%.0f bp  "
            "mu_log=%.3f [%.3f, %.3f]  sigma_log=%.3f  rho=%.4f  "
            "bins_used=%d  signal_reliable=%s",
            result_is_full.inference,
            result_is_full.median, result_is_full.mean,
            result_is_full.mu_log, *result_is_full.mu_log_ci,
            result_is_full.sigma_log, result_is_full.rho,
            result_is_full.n_bins_used, result_is_full.signal_reliable,
        )

    # Deconvolve to bp-space if density is available and NUTS samples exist.
    bp_estimate: FragmentLengthBPEstimate | None = None
    raw_samples_for_deconv = (
        result_is_full.raw_samples  # type: ignore[possibly-undefined]
        if insert_size_inference != "map" else None
    )
    if density is not None and density > 0 and raw_samples_for_deconv is not None:
        bp_estimate = _deconvolve_to_bp_space(raw_samples_for_deconv, density)
        log.info(
            "BP-space insert size: median=%.1f bp (95%% CI: %.1f\u2013%.1f), "
            "\u03c3_bp=%.3f, noise_fraction=%.1f%%",
            bp_estimate.median_bp,
            bp_estimate.median_bp_ci[0], bp_estimate.median_bp_ci[1],
            bp_estimate.sigma_bp,
            bp_estimate.noise_fraction * 100,
        )
        is_dict["bp_space"] = {
            "mu_bp": bp_estimate.mu_bp,
            "sigma_bp": bp_estimate.sigma_bp,
            "mu_bp_ci": list(bp_estimate.mu_bp_ci),
            "sigma_bp_ci": list(bp_estimate.sigma_bp_ci),
            "median_bp": bp_estimate.median_bp,
            "mean_bp": bp_estimate.mean_bp,
            "median_bp_ci": list(bp_estimate.median_bp_ci),
            "noise_fraction": bp_estimate.noise_fraction,
            "density": bp_estimate.density,
        }
    elif density is None and insert_size_inference != "map":
        log.info(
            "No --density provided; skipping bp-space deconvolution. "
            "Pass --density to get bp-space insert size estimates.",
        )

    if insert_size_json:
        insert_size_json.write_text(json.dumps(is_dict, indent=2) + "\n")
        log.info("Insert size estimate written to %s", insert_size_json)
    for env, _db in pe_shards:
        env.close()  # type: ignore[union-attr]


def _run_read_length_estimation(
    idx: _IndexHandles,
    *,
    read_length_reads: int,
    read_length_json: Path | None,
    rng: random.Random,
) -> float | None:
    """Estimate read length distribution via MAP (minimizer-count units).

    Returns:
        Estimated median read length in minimizer counts, or None if skipped.
    """
    if not _PYRO_AVAILABLE:
        log.warning("pyro-ppl and torch are not installed; skipping read length estimation.")
        return None

    log.info("Estimating read length from %d sampled reads (minimizer-count units)...", read_length_reads)
    rl_result = estimate_read_length_map(
        idx.reads_shards,
        n_reads=read_length_reads,
        rng=rng,
    )
    log.info(
        "Read length MAP: median=%.1f bp  mean=%.1f bp  "
        "mu_log=%.3f  sigma_log=%.3f  n_reads=%d  signal_reliable=%s",
        rl_result.median, rl_result.mean,
        rl_result.mu_log, rl_result.sigma_log,
        rl_result.n_reads, rl_result.signal_reliable,
    )
    if read_length_json:
        rl_dict: dict[str, object] = {
            "mu_log": rl_result.mu_log,
            "sigma_log": rl_result.sigma_log,
            "median_bp": rl_result.median,
            "mean_bp": rl_result.mean,
            "n_reads": rl_result.n_reads,
            "signal_reliable": rl_result.signal_reliable,
            "loss_final": rl_result.loss_final,
        }
        read_length_json.write_text(json.dumps(rl_dict, indent=2) + "\n")
        log.info("Read length estimate written to %s", read_length_json)
    return rl_result.median


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    prefix: Annotated[
        Path,
        typer.Argument(help="rust-mdbg output prefix (same value as --prefix)."),
    ],
    k: Annotated[
        int,
        typer.Option("--k", "-k", help="Assembly k: consecutive minimizers per graph node."),
    ] = 7,
    n_paths: Annotated[
        int,
        typer.Option("--n-paths", "-n", help="Number of paths to sample."),
    ] = 100,
    max_path_mers: Annotated[
        int,
        typer.Option(help="Maximum minimizers per path."),
    ] = 5000,
    min_support: Annotated[
        int,
        typer.Option(help="Minimum reads supporting an extension step."),
    ] = 2,
    max_reads_per_kmer: Annotated[
        int,
        typer.Option(help="Maximum reads to examine per k-mer lookup."),
    ] = 200,
    n_filter_mers: Annotated[
        int,
        typer.Option(
            help=(
                "Number of overlap minimizers to intersect before verifying the full overlap. "
                "0 means use k-1 (the complete overlap, default)."
            ),
        ),
    ] = 0,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for reproducibility."),
    ] = 42,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSONL file (default: stdout)."),
    ] = None,
    no_bloom: Annotated[
        bool,
        typer.Option("--no-bloom", help="Disable bloom filter pre-filtering."),
    ] = False,
    estimate_insert_size: Annotated[
        bool,
        typer.Option(
            "--estimate-insert-size/--no-estimate-insert-size",
            help="Run insert size estimation after path sampling (requires PE combo LMDB).",
        ),
    ] = True,
    insert_size_bins: Annotated[
        str,
        typer.Option(
            help="Comma-separated bp bin edges for insert size estimation.",
        ),
    ] = _DEFAULT_INSERT_BINS,
    pe_combo_density: Annotated[
        float,
        typer.Option(
            help=(
                "PE combo thinning density for path sketches. "
                "Must match --pe-combo-density used when building the PE combo LMDB."
            ),
        ),
    ] = 0.05,
    insert_size_inference: Annotated[
        str,
        typer.Option(
            help="Insert size inference backend: 'nuts' (MCMC), 'map' (fast MAP), or 'advi'.",
        ),
    ] = "nuts",
    insert_size_paths: Annotated[
        int,
        typer.Option(help="Maximum number of sampled paths to use for insert size estimation."),
    ] = 500,
    min_path_hashes_per_bin: Annotated[
        int,
        typer.Option(help="Minimum combo hashes in a path-bin to include it in inference."),
    ] = 50,
    insert_size_json: Annotated[
        Path | None,
        typer.Option(
            "--insert-size-json",
            help="Write insert size estimate to this JSON file (default: log only).",
        ),
    ] = None,
    estimate_read_length: Annotated[
        bool,
        typer.Option(
            "--estimate-read-length/--no-estimate-read-length",
            help="Estimate read length distribution via MAP (requires bp positions in reads LMDB).",
        ),
    ] = True,
    read_length_reads: Annotated[
        int,
        typer.Option(help="Number of reads to sample for read length estimation."),
    ] = 1000,
    read_length_json: Annotated[
        Path | None,
        typer.Option(
            "--read-length-json",
            help="Write read length MAP estimate to this JSON file (default: log only).",
        ),
    ] = None,
    read_length_bp: Annotated[
        float | None,
        typer.Option(
            "--read-length",
            help=(
                "Known read length in bp for the insert size containment kernel. "
                "If omitted, estimated automatically from the reads LMDB."
            ),
        ),
    ] = None,
    combo_max_distance: Annotated[
        int | None,
        typer.Option(
            "--combo-max-distance",
            help=(
                "The --combo-max-distance value used in rust-mdbg. "
                "Determines whether to use rectangular or triangle containment kernel."
            ),
        ),
    ] = None,
    density: Annotated[
        float | None,
        typer.Option(
            help=(
                "Minimizer density (minimizers per bp) used by rust-mdbg. "
                "Required for bp-space insert-size deconvolution and kernel unit conversion."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Sample assembly-free de Bruijn paths from metagenomic reads.

    Reads the kminmer index ({PREFIX}.kminmer_index.lmdb) and per-read
    minimizer index ({PREFIX}.index.lmdb) produced by rust-mdbg to sample
    paths through the implicit minimizer-space de Bruijn graph.

    At each extension step, the k-mer at the current path tip is looked up in
    the kminmer index to find supporting reads.  Those reads are then examined
    to find which minimizer immediately follows the tip (or precedes its
    reverse complement), collecting a vote over all supporting reads.  An
    extension is accepted only when a single unambiguous candidate clears
    min_support votes; otherwise the path terminates.  The bloom filter
    ({PREFIX}.kminmer_bloom.bin) is used for fast pre-filtering before LMDB
    lookups.

    Each output line is a JSON object::

        {
          "minimizer_ids": [int, ...],       # minimizer hash IDs along the path
          "distances":     [int, ...],       # median bp between consecutive mers
          "support":       [int, ...]        # read-support count per extension
        }

    ``distances[i]`` is the median bp gap between ``minimizer_ids[i]`` and
    ``minimizer_ids[i+1]`` as observed across supporting reads.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    rng = random.Random(seed)
    idx = _open_all_indexes(prefix, no_bloom)

    out_fp = open(output, "w") if output else sys.stdout
    sampled_results, n_written, attempts = _run_path_sampling(
        idx,
        k=k,
        n_paths=n_paths,
        max_path_mers=max_path_mers,
        min_support=min_support,
        max_reads_per_kmer=max_reads_per_kmer,
        rng=rng,
        out_fp=out_fp,
        n_filter_mers=n_filter_mers if n_filter_mers > 0 else None,
    )
    if output:
        out_fp.close()
    log.info("Wrote %d paths in %d attempts", n_written, attempts)

    # Estimate read length first — insert size estimation needs it.
    estimated_rl: float | None = None
    if estimate_read_length:
        estimated_rl = _run_read_length_estimation(
            idx,
            read_length_reads=read_length_reads,
            read_length_json=read_length_json,
            rng=rng,
        )

    # Resolve the read length for the insert size kernel.
    rl_for_insert = read_length_bp if read_length_bp is not None else estimated_rl
    if rl_for_insert is None:
        rl_for_insert = 150.0
        log.warning(
            "No read length provided or estimated; defaulting to %.0f bp "
            "for insert size containment kernel.",
            rl_for_insert,
        )

    if estimate_insert_size and sampled_results:
        _run_insert_size_estimation(
            prefix,
            sampled_results,
            insert_size_bins=insert_size_bins,
            pe_combo_density=pe_combo_density,
            insert_size_inference=insert_size_inference,
            insert_size_paths=insert_size_paths,
            min_path_hashes_per_bin=min_path_hashes_per_bin,
            insert_size_json=insert_size_json,
            read_length_bp=rl_for_insert,
            combo_max_distance=combo_max_distance,
            density=density,
        )


if __name__ == "__main__":
    app()
