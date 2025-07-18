import json
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple
from jonckheere_terpstra import jonckheere_terpstra_test

_R_TESTING_DATA_DIR = Path("R_generation")
_PYTHON_TESTING_DATA_DIR = Path("python_generation")

_PERMUTATION_TESTING_DATA = _R_TESTING_DATA_DIR / "jonckheere_test_results_PMCMRplus.json"
_APPROXIMATE_TESTING_DATA = _R_TESTING_DATA_DIR / "jonckheere_test_results_clinfun.json"
_APPROXIMATE_REGRESSION_PACK_TESTING_DATA = _PYTHON_TESTING_DATA_DIR / "jonckheere_test_results_regression_pack.json"
_EXACT_REGRESSION_PACK_TESTING_DATA = _PYTHON_TESTING_DATA_DIR / "exact_jonckheere_test_results_regression_pack.json"

# Load test data
with open(_PERMUTATION_TESTING_DATA, 'r') as f:
    permutation_benchmarks = json.load(f)

with open(_APPROXIMATE_TESTING_DATA, 'r') as f:
    approximate_benchmarks = json.load(f)

with open(_APPROXIMATE_REGRESSION_PACK_TESTING_DATA, 'r') as f:
    approximate_rp_benchmarks = json.load(f)

with open(_EXACT_REGRESSION_PACK_TESTING_DATA, 'r') as f:
    exact_rp_benchmarks = json.load(f)


def extract_run(record: dict) -> Tuple[int, float, float]:
    """
    Utility function for running the tests

    :param record: dictionary of benchmark data
    :return:
    """
    x = np.array(record["x"])
    g = np.array(record["g"])
    alternative = record["alt"]
    nperm = record.get("nperm")
    if isinstance(nperm, int):
        nperm = nperm
    else:
        nperm = None
    continuity = record.get("continuity")
    if isinstance(continuity, int):
        continuity = continuity
    else:
        continuity = True
    statistic, p_value, zstat = jonckheere_terpstra_test(x, g,
                                                         alternative=alternative,
                                                         nperm=nperm,
                                                         continuity=continuity
                                                         )
    return statistic, p_value, zstat


def is_close(a, b, tol=1e-4):
    return abs(a - b) < tol


@pytest.mark.parametrize("record",
                         permutation_benchmarks,
                         ids=[f"case_{i}" for i, _ in enumerate(permutation_benchmarks)])
def test_jonckheere_permutation(record: dict):
    """
    Test the permutation cases from PCMRPlus

    :param record: Dictionary of generated testing data
    :return:
    """
    statistic, p_value, zstat = extract_run(record)
    assert is_close(statistic, record["statistic"], tol=0.1), (
        f"Statistic mismatch in {record}: expected {record['statistic']}, got {statistic}"
    )
    assert is_close(p_value, record["p_value"], tol=0.1), (
        f"P-value mismatch in {record}: expected {record['p_value']}, got {p_value}"
    )
    assert (0.05 > p_value) == record["significant"], (
        f"Significance differs in {record}: expected {record['significant']}, got {0.05 > p_value}"
    )


@pytest.mark.parametrize("record",
                         approximate_benchmarks,
                         ids=[f"case_{i}" for i, _ in enumerate(approximate_benchmarks)])
def test_jonckheere_approximate(record: dict):
    """
    Test the approximate cases from clinfun

    :param record: Dictionary of generated testing data
    :return:
    """
    statistic, p_value, zstat = extract_run(record)
    assert is_close(statistic, record["statistic"], tol=1.1), (
        f"Statistic mismatch in {record}: expected {record['statistic']}, got {statistic}"
    )
    assert is_close(p_value, record["p_value"], tol=0.1), (
        f"P-value mismatch in {record}: expected {record['p_value']}, got {p_value}"
    )
    assert is_close(zstat, record["zstat"], tol=0.1), (
        f"Z-stat mismatch in {record}: expected {record['zstat']}, got {zstat}"
    )
    assert (0.05 > p_value) == record["significant"], (
        f"Significance differs in {record}: expected {record['significant']}, got {0.05 > p_value}"
    )


@pytest.mark.parametrize("record",
                         approximate_rp_benchmarks,
                         ids=[f"case_{i}" for i, _ in enumerate(approximate_rp_benchmarks)])
def test_jonckheere_approximate_rp(record: dict):
    """
    Test the approximate cases from regressionpack 1.0.5

    :param record: Dictionary of generated testing data
    :return:
    """
    statistic, p_value, zstat = extract_run(record)
    assert is_close(statistic, record["statistic"], tol=0.1), (
        f"Statistic mismatch in {record}: expected {record['statistic']}, got {statistic}"
    )
    assert is_close(p_value, record["p_value"], tol=0.1), (
        f"P-value mismatch in {record}: expected {record['p_value']}, got {p_value}"
    )
    assert is_close(zstat, record["zstat"], tol=0.1), (
        f"Z-stat mismatch in {record}: expected {record['zstat']}, got {zstat}"
    )
    assert (0.05 > p_value) == record["significant"], (
        f"Significance differs in {record}: expected {record['significant']}, got {0.05 > p_value}"
    )


@pytest.mark.parametrize("record",
                         exact_rp_benchmarks,
                         ids=[f"case_{i}" for i, _ in enumerate(exact_rp_benchmarks)])
def test_jonckheere_exact_rp(record: dict):
    """

    :param record:
    :return:
    """
    statistic, p_value, zstat = extract_run(record)
    assert is_close(statistic, record["statistic"], tol=0.1), (
        f"Statistic mismatch in {record}: expected {record['statistic']}, got {statistic}"
    )
    assert is_close(p_value, record["p_value"], tol=0.1), (
        f"P-value mismatch in {record}: expected {record['p_value']}, got {p_value}"
    )
    assert (0.05 > p_value) == record["significant"], (
        f"Significance differs in {record}: expected {record['significant']}, got {0.05 > p_value}"
    )
