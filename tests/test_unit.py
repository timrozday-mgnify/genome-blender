"""Unit tests for individual functions in generate_reads."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from generate_reads import (
    Fragment,
    LogLinearCalibration,
    PhredCalibration,
    Read,
    ReadBatch,
    SigmoidCalibration,
    _build_emission_logits,
    _build_sticky_transitions,
    _gc_fraction,
    _lognormal_params_from_mean_variance,
    _load_yaml_config,
    _nb_params_from_mean_variance,
    _reverse_complement,
    apply_errors_to_sequence,
    batch_sample_quality_scores,
    build_quality_calibration,
    default_illumina_profile,
    default_nanopore_profile,
    default_pacbio_profile,
)


# ------------------------------------------------------------------ #
# _gc_fraction
# ------------------------------------------------------------------ #

class TestGcFraction:
    """Tests for _gc_fraction."""

    def test_empty_string(self) -> None:
        assert _gc_fraction("") == 0.5

    def test_all_gc(self) -> None:
        assert _gc_fraction("GGCC") == pytest.approx(1.0)

    def test_all_at(self) -> None:
        assert _gc_fraction("AATT") == pytest.approx(0.0)

    def test_even_mix(self) -> None:
        assert _gc_fraction("ACGT") == pytest.approx(0.5)

    def test_case_insensitive(self) -> None:
        assert _gc_fraction("acgt") == pytest.approx(0.5)


# ------------------------------------------------------------------ #
# _reverse_complement
# ------------------------------------------------------------------ #

class TestReverseComplement:
    """Tests for _reverse_complement."""

    def test_simple(self) -> None:
        assert _reverse_complement("ACGT") == "ACGT"

    def test_asymmetric(self) -> None:
        assert _reverse_complement("AAAC") == "GTTT"

    def test_single_base(self) -> None:
        assert _reverse_complement("A") == "T"

    def test_roundtrip(self) -> None:
        seq = "ACGTACGTAC"
        assert _reverse_complement(_reverse_complement(seq)) == seq


# ------------------------------------------------------------------ #
# _nb_params_from_mean_variance
# ------------------------------------------------------------------ #

class TestNbParams:
    """Tests for _nb_params_from_mean_variance."""

    def test_variance_greater_than_mean(self) -> None:
        dist_name, params = _nb_params_from_mean_variance(300.0, 1000.0)
        assert dist_name == "nb"
        assert "total_count" in params
        assert "probs" in params
        assert params["total_count"] > 0
        assert 0 < params["probs"] < 1

    def test_variance_equals_mean_falls_back_to_poisson(self) -> None:
        dist_name, params = _nb_params_from_mean_variance(300.0, 300.0)
        assert dist_name == "poisson"
        assert params["rate"] == 300.0

    def test_variance_less_than_mean_falls_back_to_poisson(self) -> None:
        dist_name, params = _nb_params_from_mean_variance(300.0, 100.0)
        assert dist_name == "poisson"
        assert params["rate"] == 300.0

    def test_nb_mean_recoverable(self) -> None:
        """Pyro NB mean = r * p / (1 - p)."""
        mean, var = 300.0, 1000.0
        _, params = _nb_params_from_mean_variance(mean, var)
        r = params["total_count"]
        p = params["probs"]
        recovered_mean = r * p / (1 - p)
        assert recovered_mean == pytest.approx(mean, rel=1e-6)


# ------------------------------------------------------------------ #
# _lognormal_params_from_mean_variance
# ------------------------------------------------------------------ #

class TestLognormalParams:
    """Tests for _lognormal_params_from_mean_variance."""

    def test_positive_output(self) -> None:
        mu, sigma = _lognormal_params_from_mean_variance(150.0, 10.0)
        assert sigma > 0

    def test_mean_recoverable(self) -> None:
        """E[X] = exp(mu + sigma^2/2)."""
        target_mean = 150.0
        mu, sigma = _lognormal_params_from_mean_variance(
            target_mean, 10.0,
        )
        recovered = math.exp(mu + sigma**2 / 2)
        assert recovered == pytest.approx(target_mean, rel=1e-6)


# ------------------------------------------------------------------ #
# _build_emission_logits
# ------------------------------------------------------------------ #

class TestBuildEmissionLogits:
    """Tests for _build_emission_logits."""

    def test_shape(self) -> None:
        logits = _build_emission_logits(
            3, [10.0, 20.0, 30.0], [2.0, 2.0, 2.0],
        )
        assert logits.shape == (3, 94)

    def test_peak_is_maximum(self) -> None:
        logits = _build_emission_logits(
            1, [30.0], [3.0],
        )
        assert logits[0].argmax().item() == 30

    def test_custom_num_quality_values(self) -> None:
        logits = _build_emission_logits(
            1, [5.0], [2.0], num_quality_values=50,
        )
        assert logits.shape == (1, 50)


# ------------------------------------------------------------------ #
# _build_sticky_transitions
# ------------------------------------------------------------------ #

class TestBuildStickyTransitions:
    """Tests for _build_sticky_transitions."""

    def test_shape(self) -> None:
        logits = _build_sticky_transitions(4)
        assert logits.shape == (4, 4)

    def test_diagonal_dominant(self) -> None:
        logits = _build_sticky_transitions(3, self_logit=5.0)
        for s in range(3):
            assert logits[s, s].item() == 5.0
            for t in range(3):
                assert logits[s, s] >= logits[s, t]

    def test_neighbour_logits(self) -> None:
        logits = _build_sticky_transitions(
            3, self_logit=3.0, neighbour_logit=1.0,
        )
        assert logits[0, 1].item() == 1.0
        assert logits[1, 0].item() == 1.0
        assert logits[1, 2].item() == 1.0


# ------------------------------------------------------------------ #
# Calibration models
# ------------------------------------------------------------------ #

class TestPhredCalibration:
    """Tests for PhredCalibration."""

    def test_q30(self) -> None:
        cal = PhredCalibration()
        p = cal(torch.tensor([30.0]))
        assert p.item() == pytest.approx(1e-3, rel=1e-4)

    def test_q0(self) -> None:
        cal = PhredCalibration()
        p = cal(torch.tensor([0.0]))
        assert p.item() == pytest.approx(1.0, rel=1e-4)

    def test_monotonic_decrease(self) -> None:
        cal = PhredCalibration()
        qs = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])
        ps = cal(qs)
        for i in range(len(ps) - 1):
            assert ps[i] > ps[i + 1]


class TestLogLinearCalibration:
    """Tests for LogLinearCalibration."""

    def test_clamped_to_floor_at_high_q(self) -> None:
        cal = LogLinearCalibration(floor=1e-7)
        p = cal(torch.tensor([93.0]))
        assert p.item() >= 1e-7

    def test_clamped_to_ceiling_at_low_q(self) -> None:
        cal = LogLinearCalibration(ceiling=0.5)
        p = cal(torch.tensor([0.0]))
        assert p.item() <= 0.5


class TestSigmoidCalibration:
    """Tests for SigmoidCalibration."""

    def test_midpoint_gives_middle_value(self) -> None:
        cal = SigmoidCalibration(midpoint=15.0, floor=0.0, ceiling=1.0)
        p = cal(torch.tensor([15.0]))
        assert p.item() == pytest.approx(0.5, abs=0.01)

    def test_bounded(self) -> None:
        cal = SigmoidCalibration(floor=1e-6, ceiling=0.5)
        ps = cal(torch.arange(94, dtype=torch.float))
        assert ps.min().item() >= 1e-6
        assert ps.max().item() <= 0.5 + 1e-6


# ------------------------------------------------------------------ #
# build_quality_calibration
# ------------------------------------------------------------------ #

class TestBuildQualityCalibration:
    """Tests for build_quality_calibration."""

    def test_phred(self, rng) -> None:
        cal = build_quality_calibration("phred", 0.0, rng)
        assert isinstance(cal, PhredCalibration)

    def test_log_linear(self, rng) -> None:
        cal = build_quality_calibration("log-linear", 0.0, rng)
        assert isinstance(cal, LogLinearCalibration)

    def test_sigmoid(self, rng) -> None:
        cal = build_quality_calibration("sigmoid", 0.0, rng)
        assert isinstance(cal, SigmoidCalibration)

    def test_unknown_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            build_quality_calibration("bogus", 0.0, rng)

    def test_variability_perturbs_params(self, rng) -> None:
        cal0 = build_quality_calibration(
            "log-linear", 0.0, rng, intercept=-0.3,
        )
        rng2 = torch.Generator()
        rng2.manual_seed(42)
        cal1 = build_quality_calibration(
            "log-linear", 1.0, rng2, intercept=-0.3,
        )
        assert cal0.intercept != cal1.intercept


# ------------------------------------------------------------------ #
# Error model profiles
# ------------------------------------------------------------------ #

class TestErrorModelProfiles:
    """Tests for default error model profiles."""

    @pytest.mark.parametrize("factory", [
        default_illumina_profile,
        default_pacbio_profile,
        default_nanopore_profile,
    ])
    def test_ratios_sum_to_one(self, factory) -> None:
        p = factory()
        total = (
            p.substitution_ratio + p.insertion_ratio + p.deletion_ratio
        )
        assert total == pytest.approx(1.0)

    @pytest.mark.parametrize("factory", [
        default_illumina_profile,
        default_pacbio_profile,
        default_nanopore_profile,
    ])
    def test_tensor_shapes(self, factory) -> None:
        p = factory()
        assert p.initial_logits.shape == (p.num_states,)
        assert p.transition_logits.shape == (
            p.num_states, p.num_states,
        )
        assert p.emission_logits.shape[0] == p.num_states


# ------------------------------------------------------------------ #
# batch_sample_quality_scores
# ------------------------------------------------------------------ #

class TestBatchSampleQualityScores:
    """Tests for batch_sample_quality_scores."""

    def test_empty_input(self) -> None:
        profile = default_illumina_profile()
        assert batch_sample_quality_scores(profile, []) == []

    def test_output_lengths_match(self) -> None:
        torch.manual_seed(0)
        profile = default_illumina_profile()
        lengths = [10, 20, 15]
        result = batch_sample_quality_scores(profile, lengths)
        assert len(result) == 3
        for tensor, expected_len in zip(result, lengths):
            assert tensor.shape == (expected_len,)

    def test_values_in_range(self) -> None:
        torch.manual_seed(0)
        profile = default_illumina_profile()
        result = batch_sample_quality_scores(profile, [50])
        assert result[0].min().item() >= 0
        assert result[0].max().item() < 94


# ------------------------------------------------------------------ #
# apply_errors_to_sequence
# ------------------------------------------------------------------ #

class TestApplyErrorsToSequence:
    """Tests for apply_errors_to_sequence."""

    def test_no_errors_at_high_quality(self, rng) -> None:
        """Q93 -> P_error ~ 5e-10, virtually no errors."""
        seq = "ACGTACGTAC"
        q_scores = torch.full((10,), 93, dtype=torch.float64)
        profile = default_illumina_profile()
        new_seq, qual_str, cigar = apply_errors_to_sequence(
            seq, q_scores, profile, rng,
        )
        assert new_seq == seq
        assert len(qual_str) == len(seq)
        assert cigar == [(0, 10)]

    def test_all_errors_at_zero_quality(self) -> None:
        """Q0 -> P_error = 1.0, every position is an error."""
        rng = torch.Generator()
        rng.manual_seed(123)
        seq = "AAAAAAAAAA"
        q_scores = torch.zeros(10, dtype=torch.float64)
        profile = default_illumina_profile()
        new_seq, qual_str, cigar = apply_errors_to_sequence(
            seq, q_scores, profile, rng,
        )
        # With all errors the sequence or length should differ
        total_cigar_ops = sum(
            length for op, length in cigar if op != 0
        )
        assert total_cigar_ops > 0

    def test_quality_string_is_phred33(self, rng) -> None:
        seq = "ACGT"
        q_scores = torch.tensor([30, 30, 30, 30], dtype=torch.float64)
        profile = default_illumina_profile()
        _, qual_str, _ = apply_errors_to_sequence(
            seq, q_scores, profile, rng,
        )
        for ch in qual_str:
            assert ord(ch) >= 33

    def test_cigar_ops_are_valid(self, rng) -> None:
        seq = "ACGTACGTAC" * 5
        q_scores = torch.full((50,), 15, dtype=torch.float64)
        profile = default_illumina_profile()
        _, _, cigar = apply_errors_to_sequence(
            seq, q_scores, profile, rng,
        )
        for op, length in cigar:
            assert op in (0, 1, 2)
            assert length > 0

    def test_with_calibration(self, rng) -> None:
        seq = "ACGTACGTAC"
        q_scores = torch.full((10,), 30, dtype=torch.float64)
        profile = default_illumina_profile()
        cal = PhredCalibration()
        new_seq, qual_str, cigar = apply_errors_to_sequence(
            seq, q_scores, profile, rng, calibration=cal,
        )
        assert len(qual_str) > 0
        assert len(cigar) > 0


# ------------------------------------------------------------------ #
# ReadBatch
# ------------------------------------------------------------------ #

class TestReadBatch:
    """Tests for ReadBatch dataclass."""

    def test_single_end_is_not_paired(self) -> None:
        rb = ReadBatch(single=[Read("r1", "ACGT", "IIII")])
        assert not rb.is_paired

    def test_paired_end_is_paired(self) -> None:
        r1 = Read("r1/1", "ACGT", "IIII")
        r2 = Read("r1/2", "TGCA", "IIII")
        rb = ReadBatch(paired=[(r1, r2)])
        assert rb.is_paired


# ------------------------------------------------------------------ #
# _load_yaml_config
# ------------------------------------------------------------------ #

class TestLoadYamlConfig:
    """Tests for _load_yaml_config."""

    def test_basic_load(self, tmp_path) -> None:
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "num_reads: 100\nerror_model: illumina\n"
        )
        result = _load_yaml_config(cfg)
        assert result["num_reads"] == 100
        assert result["error_model"] == "illumina"

    def test_kebab_case_normalised(self, tmp_path) -> None:
        cfg = tmp_path / "test.yaml"
        cfg.write_text("read-length-mean: 150.0\n")
        result = _load_yaml_config(cfg)
        assert "read_length_mean" in result

    def test_non_dict_raises(self, tmp_path) -> None:
        cfg = tmp_path / "test.yaml"
        cfg.write_text("- item1\n- item2\n")
        with pytest.raises(Exception, match="mapping"):
            _load_yaml_config(cfg)
