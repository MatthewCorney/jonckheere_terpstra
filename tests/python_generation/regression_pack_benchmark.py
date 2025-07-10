import json

import numpy as np
from tests.python_generation.regressionpack import JonckheereTerpstra
from pathlib import Path
from typing import Optional

_PYTHON_TESTING_DATA_DIR = Path('tests') / "python_generation"
_APPROXIMATE_REGRESSION_PACK_TESTING_DATA = _PYTHON_TESTING_DATA_DIR / "jonckheere_test_results_regression_pack.json"
_EXACT_REGRESSION_PACK_TESTING_DATA = _PYTHON_TESTING_DATA_DIR / "exact_jonckheere_test_results_regression_pack.json"


def _build_group_data_helper(group_size: int, groups: int, slope: float, seed: Optional[int] = None):
    """
    Builds the data object required for the regressionpack approach

    :param group_size: Size of the groups
    :param groups: Number of groups
    :param slope: Slope of the data
    :param seed: Seed for randomness
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    base = np.random.rand(group_size, 1)
    trend = slope * np.arange(groups)
    data = base + trend
    return data.tolist()


def generate_approximate():
    """
    Creates the approximate benchmark file
    :return:
    """
    results = []
    for i in range(0, 2):
        for group_size in [5, 10, 20, 30]:
            for group_number in [3, 4, 5]:
                slope = np.random.uniform(-1, 1, 1)[0].item()
                data = _build_group_data_helper(group_size=group_size,
                                                groups=group_number,
                                                slope=slope,
                                                seed=i
                                                )
                jt = JonckheereTerpstra(data)

                p_value = jt.ComputeApproximateProbability()
                statistic = int(jt.J)
                zstat = float(jt.Z)
                p_value = float(p_value)
                x = []
                g = []
                for index, values in enumerate(data):
                    g.extend([index + 1 for _ in values])
                    x.extend(values)
                results.append({
                    'ties': len(set(x)) > len(x),
                    'more_than_100_obs': 100 < len(x),
                    'x': x,
                    'g': g,
                    'slope': slope,
                    'id': i,
                    'continuity': False,
                    'nperm': None,
                    'group_size': group_size,
                    'group_number': group_number,
                    'alt': 'increasing',
                    'statistic': statistic,
                    'zstat': zstat,
                    'p_value': p_value,
                    'significant': 0.05 > p_value,
                })
    with open(_APPROXIMATE_REGRESSION_PACK_TESTING_DATA, 'w') as f:
        json.dump(obj=results, fp=f)


def generate_exact():
    """
    Creates the exact benchmark file
    :return:
    """
    results = []
    for i in range(0, 10):
        for group_size in [3]:
            for group_number in [3]:
                slope = np.random.uniform(-1, 1, 1)[0].item()
                data = _build_group_data_helper(group_size=group_size,
                                                groups=group_number,
                                                slope=slope,
                                                seed=i
                                                )
                jt = JonckheereTerpstra(data)

                p_value = jt.ComputeExactProbability()
                statistic = int(jt.J)
                zstat = None
                p_value = float(p_value)
                x = []
                g = []
                for index, values in enumerate(data):
                    g.extend([index + 1 for _ in values])
                    x.extend(values)
                results.append({
                    'ties': len(set(x)) > len(x),
                    'more_than_100_obs': 100 < len(x),
                    'x': x,
                    'g': g,
                    'slope': slope,
                    'id': i,
                    'continuity': False,
                    'nperm': None,
                    'group_size': group_size,
                    'group_number': group_number,
                    'alt': 'increasing',
                    'statistic': statistic,
                    'zstat': zstat,
                    'p_value': p_value,
                    'significant': 0.05 > p_value,
                })
    with open(_EXACT_REGRESSION_PACK_TESTING_DATA, 'w') as f:
        json.dump(obj=results, fp=f)


if __name__ == '__main__':
    generate_approximate()
    generate_exact()
