"""Tests for scripts/asf_sample.py."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path

import lmdb
import numpy as np
import pytest
from typer.testing import CliRunner

# Import from the script as a module.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from asf_sample import (
    FragmentLengthEstimate,
    FragmentLengthMAP,
    PathResult,
    ReadLengthMAP,
    _BloomFilter,
    _bloom_check,
    _build_path_bin_sketches,
    _canonical_combo_hash_v,
    _decode_read,
    _fnv1a64,
    _open_bloom,
    _open_reads_lmdb,
    _sample_read_ids,
    app,
)

# These symbols are not yet implemented in asf_sample.py.
try:
    from asf_sample import _canonical, _kmer_bytes, _open_kminmer_lmdb
    _KMER_UTILS_AVAILABLE = True
except ImportError:
    _KMER_UTILS_AVAILABLE = False

try:
    from asf_sample import (
        _SuccessionIndex,
        _lookup_succession,
        _open_succession,
        _SUCC_HEADER_FMT,
        _SUCC_HEADER_SIZE,
        _SUCC_MAGIC,
    )
    _SUCCESSION_AVAILABLE = True
except ImportError:
    _SUCCESSION_AVAILABLE = False

# Conditional imports for Pyro model tests.
try:
    from asf_sample import (
        estimate_read_length_map,
    )

    _PYRO_AVAILABLE = True
except ImportError:
    _PYRO_AVAILABLE = False

# E2E real data path.
_REAL_PREFIX = Path(
    "/Users/timrozday/Documents/genome-blender_run/"
    "single_short_shallow/output/rust-mdbg/rust_mdbg_out"
)
_REAL_DATA_EXISTS = _REAL_PREFIX.with_suffix(".index.lmdb").exists()


# ---------------------------------------------------------------------------
# Synthetic LMDB fixture helpers
# ---------------------------------------------------------------------------


def _write_reads_lmdb(
    path: Path,
    reads: dict[int, list[int]],
) -> None:
    """Write a minimal reads LMDB with ``reads`` and ``meta`` sub-databases.

    Args:
        path: Directory to create.
        reads: Mapping of 1-based read ID to list of minimizer hash ints.
    """
    env = lmdb.open(str(path), max_dbs=8, map_size=256 * 1024 * 1024)
    reads_db = env.open_db(b"reads")
    meta_db = env.open_db(b"meta")
    with env.begin(write=True) as txn:
        for rid, mids in reads.items():
            key = struct.pack("<I", rid)  # u32 LE — matches new rust-mdbg format
            val = b"".join(struct.pack("<Q", m) for m in mids)
            txn.put(key, val, db=reads_db)
        txn.put(b"n_reads", struct.pack("<Q", len(reads)), db=meta_db)
        txn.put(b"read_id_width", struct.pack("<I", 4), db=meta_db)
    env.close()


def _write_kminmer_lmdb(
    path: Path,
    kmer_to_read_ids: dict[bytes, list[int]],
) -> None:
    """Write a minimal kminmer index LMDB with ``kminmers`` sub-database.

    Args:
        path: Directory to create.
        kmer_to_read_ids: Mapping of canonical kmer bytes to sorted read IDs.
    """
    env = lmdb.open(str(path), max_dbs=8, map_size=256 * 1024 * 1024)
    km_db = env.open_db(b"kminmers", dupsort=True)
    meta_db = env.open_db(b"meta")
    with env.begin(write=True) as txn:
        for kmer_key, rids in kmer_to_read_ids.items():
            for rid in rids:
                txn.put(kmer_key, struct.pack("<Q", rid), db=km_db, dupdata=True)
        txn.put(b"n_kminmers", struct.pack("<Q", len(kmer_to_read_ids)), db=meta_db)
    env.close()


def _write_bloom_filter(
    path: Path,
    kmer_bytes_list: list[bytes],
    k: int,
    lmer_len: int = 21,
    n_bits: int = 8192,
) -> None:
    """Write a minimal bloom filter file containing the given kmers."""
    n_u64 = (n_bits + 63) // 64
    bits = np.zeros(n_u64, dtype=np.uint64)
    for kb in kmer_bytes_list:
        h1 = _fnv1a64(kb)
        h2 = (((h1 << 17) | (h1 >> 47)) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        for i in range(3):
            pos = ((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % n_bits
            bits[pos >> 6] |= np.uint64(1 << (pos & 63))
    header = struct.pack("<8sQQQQQ", b"KMBLOOMB", 0, k, lmer_len, n_bits, 3)
    with open(path, "wb") as f:
        f.write(header)
        f.write(bits.tobytes())


def _write_succession_index(
    path: Path,
    k: int,
    entries: list[tuple[list[int], list[tuple[int, int]], list[tuple[int, int]]]],
) -> None:
    """Write a minimal succession index binary file.

    Args:
        path: Output file path.
        k: Number of minimizers per k-mer.
        entries: List of (canonical_key, fwd_entries, bwd_entries) where
            fwd/bwd entries are (hash, count) pairs.
            canonical_key is a list of k uint64 values.
            Entries MUST be sorted by canonical key bytes (LE byte order).
    """
    n_kmers = len(entries)
    # Build data section.
    data_parts: list[bytes] = []
    offsets: list[int] = []
    for _key, fwd, bwd in entries:
        offsets.append(len(b"".join(data_parts)))
        part = struct.pack("<HH", len(fwd), len(bwd))
        for h, cnt in fwd:
            part += struct.pack("<QI", h, cnt)
        for h, cnt in bwd:
            part += struct.pack("<QI", h, cnt)
        data_parts.append(part)
    data_blob = b"".join(data_parts)
    # Build key table (sorted by canonical key bytes).
    key_blob = b"".join(
        np.array(key, dtype="<u8").tobytes() for key, _, _ in entries
    )
    offsets_blob = np.array(offsets, dtype="<u8").tobytes()
    header = struct.pack(_SUCC_HEADER_FMT, _SUCC_MAGIC, n_kmers, k, 0)
    with open(path, "wb") as f:
        f.write(header)
        f.write(key_blob)
        f.write(offsets_blob)
        f.write(data_blob)


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no I/O
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _KMER_UTILS_AVAILABLE, reason="_canonical not yet implemented")
class TestCanonical:
    """Test _canonical (lexicographic min of forward/reversed)."""

    def test_forward_is_smaller(self) -> None:
        kmer = np.array([1, 2, 3], dtype=np.uint64)
        result = _canonical(kmer)
        np.testing.assert_array_equal(result, kmer)

    def test_reversed_is_smaller(self) -> None:
        kmer = np.array([3, 2, 1], dtype=np.uint64)
        result = _canonical(kmer)
        np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint64))

    def test_palindrome(self) -> None:
        kmer = np.array([5, 3, 5], dtype=np.uint64)
        result = _canonical(kmer)
        np.testing.assert_array_equal(result, np.array([5, 3, 5], dtype=np.uint64))

    def test_single_element(self) -> None:
        kmer = np.array([42], dtype=np.uint64)
        result = _canonical(kmer)
        np.testing.assert_array_equal(result, kmer)


@pytest.mark.skipif(not _KMER_UTILS_AVAILABLE, reason="_kmer_bytes not yet implemented")
class TestKmerBytes:
    """Test _kmer_bytes (canonical encoding to LE bytes)."""

    def test_canonical_encoding(self) -> None:
        kmer = np.array([3, 2, 1], dtype=np.uint64)
        result = _kmer_bytes(kmer)
        expected = np.array([1, 2, 3], dtype="<u8").tobytes()
        assert result == expected

    def test_determinism(self) -> None:
        kmer = np.array([10, 20, 30], dtype=np.uint64)
        assert _kmer_bytes(kmer) == _kmer_bytes(kmer)

    def test_length(self) -> None:
        kmer = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint64)
        assert len(_kmer_bytes(kmer)) == 7 * 8


class TestFnv1a64:
    """Test _fnv1a64 hash function."""

    def test_empty_bytes(self) -> None:
        result = _fnv1a64(b"")
        assert result == 14695981039346656037

    def test_known_hash_a(self) -> None:
        result = _fnv1a64(b"a")
        assert isinstance(result, int)
        assert result != _fnv1a64(b"")

    def test_different_inputs_differ(self) -> None:
        assert _fnv1a64(b"hello") != _fnv1a64(b"world")

    def test_deterministic(self) -> None:
        assert _fnv1a64(b"test") == _fnv1a64(b"test")


@pytest.mark.skipif(not _KMER_UTILS_AVAILABLE, reason="_kmer_bytes not yet implemented")
class TestBloomCheck:
    """Test _bloom_check with a synthetic bloom filter."""

    def _make_bloom(self, kmer_bytes_list: list[bytes], n_bits: int = 4096) -> _BloomFilter:
        """Build an in-memory bloom filter."""
        n_u64 = (n_bits + 63) // 64
        bits = np.zeros(n_u64, dtype=np.uint64)
        for kb in kmer_bytes_list:
            h1 = _fnv1a64(kb)
            h2 = (((h1 << 17) | (h1 >> 47)) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            for i in range(3):
                pos = ((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % n_bits
                bits[pos >> 6] |= np.uint64(1 << (pos & 63))
        return _BloomFilter(bits=bits, n_bits=n_bits, n_hash_fns=3, k=3, lmer_len=21)

    def test_present_returns_true(self) -> None:
        kb = _kmer_bytes(np.array([1, 2, 3], dtype=np.uint64))
        bloom = self._make_bloom([kb])
        assert _bloom_check(bloom, kb) is True

    def test_absent_returns_false(self) -> None:
        kb_present = _kmer_bytes(np.array([1, 2, 3], dtype=np.uint64))
        kb_absent = _kmer_bytes(np.array([99, 88, 77], dtype=np.uint64))
        bloom = self._make_bloom([kb_present])
        assert _bloom_check(bloom, kb_absent) is False


class TestDecodeRead:
    """Test _decode_read byte parsing."""

    def test_basic(self) -> None:
        mids_expected = [10, 20, 30]
        val = b"".join(struct.pack("<Q", m) for m in mids_expected)
        mids = _decode_read(val)
        np.testing.assert_array_equal(mids, np.array(mids_expected, dtype=np.uint64))

    def test_empty_input(self) -> None:
        mids = _decode_read(b"")
        assert len(mids) == 0

    def test_single_minimizer(self) -> None:
        val = struct.pack("<Q", 42)
        mids = _decode_read(val)
        assert len(mids) == 1
        assert mids[0] == 42


class TestCanonicalComboHashV:
    """Test _canonical_combo_hash_v (vectorised commutative hash)."""

    def test_commutativity(self) -> None:
        a = np.array([10, 20, 30], dtype=np.uint64)
        b = np.array([40, 50, 60], dtype=np.uint64)
        h_ab = _canonical_combo_hash_v(a, b)
        h_ba = _canonical_combo_hash_v(b, a)
        np.testing.assert_array_equal(h_ab, h_ba)

    def test_determinism(self) -> None:
        a = np.array([1, 2], dtype=np.uint64)
        b = np.array([3, 4], dtype=np.uint64)
        np.testing.assert_array_equal(
            _canonical_combo_hash_v(a, b),
            _canonical_combo_hash_v(a, b),
        )

    def test_distinct_pairs_differ(self) -> None:
        a1 = np.array([1], dtype=np.uint64)
        b1 = np.array([2], dtype=np.uint64)
        a2 = np.array([1], dtype=np.uint64)
        b2 = np.array([3], dtype=np.uint64)
        assert _canonical_combo_hash_v(a1, b1)[0] != _canonical_combo_hash_v(a2, b2)[0]


class TestDataclasses:
    """Test dataclass construction and defaults."""

    def test_path_result(self) -> None:
        pr = PathResult(minimizer_ids=[1, 2, 3], distances=[10, 20], support=[5, 5])
        assert len(pr.minimizer_ids) == 3
        assert len(pr.distances) == 2

    def test_fragment_length_estimate(self) -> None:
        fle = FragmentLengthEstimate(
            mu_log=7.0, sigma_log=0.5,
            mu_log_ci=(6.5, 7.5), sigma_log_ci=(0.3, 0.7),
            rho=0.1, norm=0.5, median=1096.6, mean=1234.5,
            n_bins_used=5, inference="nuts",
        )
        assert fle.signal_reliable is True

    def test_fragment_length_map(self) -> None:
        flm = FragmentLengthMAP(
            mu_log=7.0, sigma_log=0.5, rho=0.1, norm=0.5,
            median=1096.6, mean=1234.5, n_bins_used=5,
        )
        assert flm.signal_reliable is True
        assert math.isnan(flm.loss_final)

    def test_read_length_map(self) -> None:
        rlm = ReadLengthMAP(
            mu_log=4.9, sigma_log=0.1, median=134.3, mean=135.0, n_reads=100,
        )
        assert rlm.signal_reliable is True
        assert math.isnan(rlm.loss_final)


# ---------------------------------------------------------------------------
# Integration tests — LMDB + index I/O
# ---------------------------------------------------------------------------


class TestOpenReadsLmdb:
    """Test _open_reads_lmdb round-trip with synthetic data."""

    def test_round_trip(self, tmp_path: Path) -> None:
        lmdb_dir = tmp_path / "test.index.lmdb"
        _write_reads_lmdb(lmdb_dir, {1: [10, 20, 30], 2: [40, 50]})
        prefix = tmp_path / "test"
        shards, read_id_width = _open_reads_lmdb(prefix)
        assert len(shards) >= 1
        assert read_id_width == 4
        env, reads_db, _meta_db = shards[0]
        with env.begin() as txn:
            val = txn.get(struct.pack("<I", 1), db=reads_db)
        assert val is not None
        mids = _decode_read(val)
        np.testing.assert_array_equal(mids, [10, 20, 30])
        env.close()


@pytest.mark.skipif(not _KMER_UTILS_AVAILABLE, reason="_open_kminmer_lmdb not yet implemented")
class TestOpenKminmerLmdb:
    """Test _open_kminmer_lmdb with synthetic data."""

    def test_round_trip(self, tmp_path: Path) -> None:
        lmdb_dir = tmp_path / "test.kminmer_index.lmdb"
        kmer_key = np.array([1, 2, 3], dtype="<u8").tobytes()
        _write_kminmer_lmdb(lmdb_dir, {kmer_key: [1, 2, 3]})
        prefix = tmp_path / "test"
        env, km_db, meta_db = _open_kminmer_lmdb(prefix)
        with env.begin() as txn:
            cursor = txn.cursor(db=km_db)
            assert cursor.set_key(kmer_key)
            rids = [struct.unpack("<Q", v)[0] for v in cursor.iternext_dup()]
        assert rids == [1, 2, 3]
        env.close()


@pytest.mark.skipif(not _KMER_UTILS_AVAILABLE, reason="_kmer_bytes not yet implemented")
class TestOpenBloom:
    """Test _open_bloom with synthetic bloom filter files."""

    def test_valid_file(self, tmp_path: Path) -> None:
        bloom_path = tmp_path / "test.kminmer_bloom.bin"
        kb = _kmer_bytes(np.array([1, 2, 3], dtype=np.uint64))
        _write_bloom_filter(bloom_path, [kb], k=3, n_bits=4096)
        bloom = _open_bloom(bloom_path)
        assert bloom.k == 3
        assert bloom.n_bits == 4096
        assert _bloom_check(bloom, kb) is True

    def test_invalid_magic_raises(self, tmp_path: Path) -> None:
        bloom_path = tmp_path / "bad.bin"
        header = struct.pack("<8sQQQQQ", b"NOTBLOOM", 0, 3, 21, 1024, 3)
        bits = np.zeros(16, dtype=np.uint64)
        with open(bloom_path, "wb") as f:
            f.write(header)
            f.write(bits.tobytes())
        with pytest.raises(ValueError, match="bad magic"):
            _open_bloom(bloom_path)


@pytest.mark.skipif(not _SUCCESSION_AVAILABLE, reason="Succession index reader not yet implemented")
class TestOpenSuccession:
    """Test _open_succession with synthetic data."""

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        prefix = tmp_path / "nonexistent"
        result = _open_succession(prefix)
        assert result is None

    def test_valid_file(self, tmp_path: Path) -> None:
        prefix = tmp_path / "test"
        succ_path = Path(f"{prefix}.kminmer_succession.bin")
        # Single canonical k-mer [1, 2, 3] with one forward successor.
        _write_succession_index(
            succ_path, k=3,
            entries=[([1, 2, 3], [(99, 5)], [])],
        )
        idx = _open_succession(prefix)
        assert idx is not None
        assert idx.k == 3
        assert idx.n_kmers == 1

    def test_invalid_magic_raises(self, tmp_path: Path) -> None:
        prefix = tmp_path / "test"
        succ_path = Path(f"{prefix}.kminmer_succession.bin")
        header = struct.pack(_SUCC_HEADER_FMT, b"BADMAGIC", 0, 3, 0)
        with open(succ_path, "wb") as f:
            f.write(header)
        with pytest.raises(ValueError, match="bad magic"):
            _open_succession(prefix)


@pytest.mark.skipif(not _SUCCESSION_AVAILABLE, reason="Succession index reader not yet implemented")
class TestLookupSuccession:
    """Test _lookup_succession with a synthetic succession index."""

    @pytest.fixture()
    def idx(self, tmp_path: Path) -> _SuccessionIndex:
        """Build a succession index with one canonical k-mer [1, 2, 3].

        Forward successors: hash=99 count=5 dist=100, hash=88 count=3 dist=200
        Backward predecessors: hash=77 count=4 dist=150
        """
        prefix = tmp_path / "test"
        succ_path = Path(f"{prefix}.kminmer_succession.bin")
        _write_succession_index(
            succ_path, k=3,
            entries=[
                ([1, 2, 3], [(99, 5), (88, 3)], [(77, 4)]),
            ],
        )
        result = _open_succession(prefix)
        assert result is not None
        return result

    def test_forward_lookup(self, idx: _SuccessionIndex) -> None:
        tip = np.array([1, 2, 3], dtype=np.uint64)
        counts = _lookup_succession(idx, tip, successor=True)
        assert counts[99] == 5
        assert counts[88] == 3

    def test_backward_lookup(self, idx: _SuccessionIndex) -> None:
        tip = np.array([1, 2, 3], dtype=np.uint64)
        counts = _lookup_succession(idx, tip, successor=False)
        assert counts[77] == 4

    def test_not_found(self, idx: _SuccessionIndex) -> None:
        tip = np.array([99, 98, 97], dtype=np.uint64)
        counts = _lookup_succession(idx, tip, successor=True)
        assert len(counts) == 0

    def test_orientation_flip(self, idx: _SuccessionIndex) -> None:
        """Reversed tip [3, 2, 1] canonicalises to [1, 2, 3] but flips fwd/bwd."""
        tip = np.array([3, 2, 1], dtype=np.uint64)
        # successor=True with reversed tip → reads backward entries.
        counts = _lookup_succession(idx, tip, successor=True)
        assert counts[77] == 4
        # successor=False with reversed tip → reads forward entries.
        counts = _lookup_succession(idx, tip, successor=False)
        assert counts[99] == 5


class TestSampleReadIds:
    """Test _sample_read_ids with synthetic reads LMDB."""

    def test_basic_sampling(self, tmp_path: Path) -> None:
        lmdb_dir = tmp_path / "test.index.lmdb"
        reads = {i: [i * 10] for i in range(1, 11)}
        _write_reads_lmdb(lmdb_dir, reads)
        prefix = tmp_path / "test"
        shards, _width = _open_reads_lmdb(prefix)
        rng = random.Random(42)
        txns_dbs_metas = [
            (env.begin(), db, meta_db) for env, db, meta_db in shards
        ]
        ids = _sample_read_ids(txns_dbs_metas, n=5, rng=rng)
        assert len(ids) == 5
        assert all(1 <= rid <= 10 for rid in ids)
        for env, _, _ in shards:
            env.close()


class TestBuildPathBinSketches:
    """Test _build_path_bin_sketches with synthetic path results."""

    def test_single_path_single_bin(self) -> None:
        result = PathResult(
            minimizer_ids=[10, 20, 30, 40],
            distances=[100, 200, 300],
            support=[5, 5, 5],
        )
        bin_distances = [0.0, 1000.0]
        sketches = _build_path_bin_sketches([result], bin_distances, combo_density=0.99)
        assert len(sketches) == 1
        assert len(sketches[0]) == 1  # 1 bin

    def test_empty_path_list(self) -> None:
        sketches = _build_path_bin_sketches([], [0.0, 1000.0], combo_density=0.99)
        assert len(sketches) == 0

    def test_multi_bin(self) -> None:
        result = PathResult(
            minimizer_ids=[10, 20, 30, 40, 50],
            distances=[50, 100, 200, 400],
            support=[3, 3, 3, 3],
        )
        bin_distances = [0.0, 100.0, 500.0]
        sketches = _build_path_bin_sketches([result], bin_distances, combo_density=0.99)
        assert len(sketches) == 1
        assert len(sketches[0]) == 2  # 2 bins


# ---------------------------------------------------------------------------
# Pyro model tests (skip if not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PYRO_AVAILABLE, reason="pyro-ppl not installed")
class TestReadLengthMAP:
    """Test estimate_read_length_map with synthetic reads LMDB."""

    def test_basic_estimation(self, tmp_path: Path) -> None:
        """Reads with ~10 minimizers should give median near 10 (minimizer-count)."""
        lmdb_dir = tmp_path / "test.index.lmdb"
        reads: dict[int, list[int]] = {}
        rng = random.Random(42)
        for rid in range(1, 201):
            n_mers = rng.randint(8, 15)
            reads[rid] = [rng.randint(1, 10**6) for _ in range(n_mers)]
        _write_reads_lmdb(lmdb_dir, reads)
        prefix = tmp_path / "test"
        shards, _width = _open_reads_lmdb(prefix)
        result = estimate_read_length_map(
            shards,
            n_reads=100, rng=random.Random(123),
        )
        assert result.median > 0
        assert result.mean > 0
        assert result.n_reads > 0
        for env, _, _ in shards:
            env.close()


# ---------------------------------------------------------------------------
# E2E tests — CLI via CliRunner, real data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _REAL_DATA_EXISTS, reason="Real test data not available")
class TestMainCLI:
    """End-to-end tests using real rust-mdbg output."""

    runner = CliRunner()

    def test_basic_sampling(self, tmp_path: Path) -> None:
        output_file = tmp_path / "paths.jsonl"
        result = self.runner.invoke(app, [
            str(_REAL_PREFIX),
            "--k", "7",
            "--n-paths", "5",
            "--max-path-mers", "500",
            "--seed", "42",
            "--output", str(output_file),
            "--no-estimate-insert-size",
            "--no-estimate-read-length",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "minimizer_ids" in obj
            assert "distances" in obj
            assert "support" in obj

    def test_path_lengths_exceed_k(self, tmp_path: Path) -> None:
        output_file = tmp_path / "paths.jsonl"
        k = 7
        result = self.runner.invoke(app, [
            str(_REAL_PREFIX),
            "--k", str(k),
            "--n-paths", "10",
            "--seed", "42",
            "--output", str(output_file),
            "--no-estimate-insert-size",
            "--no-estimate-read-length",
        ])
        assert result.exit_code == 0
        for line in output_file.read_text().strip().split("\n"):
            obj = json.loads(line)
            assert len(obj["minimizer_ids"]) > k

    def test_read_length_json(self, tmp_path: Path) -> None:
        output_file = tmp_path / "paths.jsonl"
        rl_json = tmp_path / "read_length.json"
        result = self.runner.invoke(app, [
            str(_REAL_PREFIX),
            "--k", "7",
            "--n-paths", "3",
            "--seed", "42",
            "--output", str(output_file),
            "--no-estimate-insert-size",
            "--estimate-read-length",
            "--read-length-reads", "100",
            "--read-length-json", str(rl_json),
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        rl_data = json.loads(rl_json.read_text())
        assert rl_data["median_bp"] > 0
        assert rl_data["mean_bp"] > 0

    def test_reproducible(self, tmp_path: Path) -> None:
        outputs = []
        for i in range(2):
            output_file = tmp_path / f"paths_{i}.jsonl"
            result = self.runner.invoke(app, [
                str(_REAL_PREFIX),
                "--k", "7",
                "--n-paths", "5",
                "--seed", "42",
                "--output", str(output_file),
                "--no-estimate-insert-size",
                "--no-estimate-read-length",
            ])
            assert result.exit_code == 0
            outputs.append(output_file.read_text())
        assert outputs[0] == outputs[1]

    def test_insert_size_map(self, tmp_path: Path) -> None:
        output_file = tmp_path / "paths.jsonl"
        is_json = tmp_path / "insert_size.json"
        result = self.runner.invoke(app, [
            str(_REAL_PREFIX),
            "--k", "7",
            "--n-paths", "20",
            "--seed", "42",
            "--output", str(output_file),
            "--estimate-insert-size",
            "--insert-size-inference", "map",
            "--insert-size-paths", "10",
            "--insert-size-bins", "0,200,400,600,800,1000,2000,4000,8000",
            "--pe-combo-density", "0.99",
            "--insert-size-json", str(is_json),
            "--no-estimate-read-length",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        is_data = json.loads(is_json.read_text())
        assert is_data["method"] == "map"
        assert is_data["median_bp"] > 0
