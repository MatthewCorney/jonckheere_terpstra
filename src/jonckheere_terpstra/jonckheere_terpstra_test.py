from typing import Union
import warnings
from scipy.stats import norm

import numpy as np
from scipy.stats import rankdata
from typing import Literal
from typing import Tuple

_ValidAlternatives = Literal['increasing', 'decreasing', 'two_sided']


def _compute_jt_statistic(x: np.ndarray[np.number], g: np.ndarray[int]) -> int:
    """
    Compute the Jonckheere–Terpstra U statistic directly
    from global ranks of x, given ordered group labels g.

    :param x: Ordered numeric response variable
    :param g: Group labels array

    :return int: The JT test statistic
    """

    # get group boundaries
    _, counts = np.unique(g, return_counts=True)
    cg: np.ndarray = np.concatenate(([0], np.cumsum(counts)))

    u = 0
    # for each ordered pair of groups (i<j)
    for i in range(len(counts) - 1):
        xi = x[cg[i]: cg[i + 1]]
        for j in range(i + 1, len(counts)):
            xj = x[cg[j]: cg[j + 1]]
            # count all xi in Xi less than each xj in Xj
            # vectorized: Xi[:,None] < Xj  → boolean matrix
            u += np.sum(xi[:, None] < xj)
    return int(u)


def _calculate_mean_variance(n: int,
                             ng: int,
                             gsize: np.ndarray,
                             cgsize: np.ndarray) -> Tuple[float, float]:
    """
    Compute the expected mean and variance of the Jonckheere–Terpstra statistic under the null hypothesis.

    These values are used in the asymptotic normal approximation to evaluate statistical significance.

    :param n: Total number of observations
    :param ng: Number of groups
    :param gsize: Array of group sizes (length ng)
    :param cgsize: Cumulative group sizes, used to compute number of observations after each group
    :return: Tuple containing:
             - Expected mean of the JT statistic under the null hypothesis (float)
             - Variance of the JT statistic under the null hypothesis (float)
    """
    jtmean = jtvar = 0
    for i in range(ng - 1):
        na = gsize[i]
        nb = n - cgsize[i + 1]
        jtmean += na * nb / 2
        jtvar += na * nb * (na + nb + 1) / 12
    return jtmean, jtvar


def _jt_approximate_pvalue(jtrsum: float,
                           jtmean: float,
                           jtvar: float,
                           alternative: _ValidAlternatives,
                           continuity: bool) -> Tuple[float, float]:
    """
    Compute the approximate (asymptotic) p-value and z-statistic for the Jonckheere–Terpstra test.

    Uses a normal approximation under the null hypothesis to compute a z-score from the observed
    JT statistic, with optional continuity correction.

    :param jtrsum: Observed Jonckheere–Terpstra statistic
    :param jtmean: Expected mean of the JT statistic under the null hypothesis
    :param jtvar: Variance of the JT statistic under the null hypothesis
    :param alternative: Direction of the trend to test. One of:
                        - "two_sided": any monotonic trend
                        - "increasing": increasing trend across groups
                        - "decreasing": decreasing trend across groups
    :param continuity: Whether to apply a continuity correction to the z-score
    :return: Tuple containing:
             - p-value (float)
             - z-statistic (float)
    """
    delta = 0.5 if continuity else 0.0
    adjustment = delta * np.sign(jtrsum - jtmean)
    zstat = (jtrsum - jtmean - adjustment) / np.sqrt(jtvar)
    if alternative == "two_sided":
        pval = 2 * norm.sf(abs(zstat))
    elif alternative == "increasing":
        pval = norm.sf(zstat)
    elif alternative == "decreasing":
        pval = norm.cdf(zstat)
    else:
        raise ValueError("Invalid alternative")
    return pval.item(), zstat.item()


def _jt_permutation_pvalue(x: np.ndarray[np.number],
                           gsize: np.ndarray[int],
                           cgsize: np.ndarray,
                           alternative: str,
                           nperm: int) -> Tuple[float, float]:
    """
    Compute a permutation-based p-value for the Jonckheere–Terpstra test.

    This function generates a null distribution of the JT statistic by resampling the data `nperm` times.
    The first value in the permutation distribution corresponds to the original, unpermuted data.

    :param x: Data values (must be numeric)
    :param gsize: Sizes of each group, in increasing order
    :param cgsize: Cumulative group sizes (used for slicing into x)
    :param alternative: One of "two_sided", "increasing", or "decreasing"
    :param nperm: Number of permutations to perform

    :return: Tuple containing:
         - p-value (float)
         - z-statistic (float)
    """
    ng = len(gsize)
    pjtrsum: np.ndarray = np.empty(nperm)
    x_perm = x.copy()

    # compute statistic on original and permuted data sequentially
    for i in range(nperm):
        # compute JT statistic for x_perm
        jtrsum = 0
        for gi in range(ng - 1):
            na = gsize[gi]
            ranks = rankdata(x_perm[cgsize[gi]:])
            jtrsum += np.sum(ranks[:na]) - na * (na + 1) / 2
        pjtrsum[i] = jtrsum
        # permute for next iteration
        x_perm = np.random.permutation(x)
    jtr0 = pjtrsum[0]
    null_stats = pjtrsum[1:]
    null_mean = np.mean(null_stats)
    null_std = np.std(null_stats, ddof=1)
    zstat = float((jtr0 - null_mean) / null_std)

    ipval = float(np.mean(pjtrsum <= jtr0))
    dpval = float(np.mean(pjtrsum >= jtr0))

    if alternative == "two_sided":
        return 2 * min([ipval, dpval, 0.5]), zstat
    elif alternative == "increasing":
        return ipval, zstat
    elif alternative == "decreasing":
        return dpval, zstat
    else:
        raise ValueError("Invalid alternative")


def jonckheere_terpstra_test(x: Union[np.ndarray, list],
                             g: Union[np.ndarray, list],
                             alternative: _ValidAlternatives = "two_sided",
                             nperm: int = 5000,
                             continuity: bool = True) -> tuple[int, float, float]:
    """
    Perform the Jonckheere–Terpstra test for ordered differences among multiple groups.

    The test evaluates whether there is a trend (increasing, decreasing, or two-sided) in the distribution
    of a numeric variable `x` across ordered groups `g`.

    :param x: Numeric response variable
    :param g: Group labels (must be orderable; numeric or ordinal)
    :param alternative: Direction of the trend to test. One of:
                        - "two_sided": test for any ordered difference
                        - "increasing": test for an increasing trend across groups
                        - "decreasing": test for a decreasing trend across groups
    :param nperm: If provided, compute p-value using `nperm` permutations instead of asymptotic or exact method
    :param continuity: if continuity correction should, be used in the approximate case

    :return: Tuple containing:
         - JT statistic (int)
         - p-value (float)
         - z-statistic (float)
    """
    if alternative not in ['increasing', 'decreasing', 'two_sided']:
        raise ValueError(f'Alternative must be one of {_ValidAlternatives} {alternative=} passed')
    x: np.ndarray[np.number] = np.asarray(x)
    g: np.ndarray[int] = np.asarray(g)
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("x must be numeric")
    if len(x) != len(g):
        raise ValueError("x and g must be the same length")
    if len(np.unique(g)) < 3:
        raise ValueError("Jonckheere–Terpstra test requires at least 3 ordered groups")
    if len(np.unique(x)) < len(x) and not nperm:
        warnings.warn("Permutation should be used if ties are present")

    finite = np.isfinite(x) & np.isfinite(g)
    x = x[finite]
    g = g[finite]

    # sort by group
    order = np.argsort(g)
    x = x[order]
    g = g[order]

    # group sizes
    _, counts = np.unique(g, return_counts=True)
    gsize: np.ndarray[int] = counts
    ng: int = len(gsize)
    cgsize: np.ndarray[int] = np.concatenate(([0], np.cumsum(gsize)))
    n: int = len(x)

    # compute mean, var, and observed statistic
    jtmean, jtvar = _calculate_mean_variance(n, ng, gsize, cgsize)

    jtrsum = _compute_jt_statistic(x, g)
    if nperm:
        pval, zstat = _jt_permutation_pvalue(x=x,
                                             gsize=gsize,
                                             cgsize=cgsize,
                                             alternative=alternative,
                                             nperm=nperm
                                             )
    else:
        pval, zstat = _jt_approximate_pvalue(jtmean=jtmean,
                                             jtvar=jtvar,
                                             jtrsum=jtrsum,
                                             alternative=alternative,
                                             continuity=continuity
                                             )
    return jtrsum, pval, zstat
#
#
# import json
#
# with open("C:\\Users\\matth\\OneDrive\\Documents\\jonckheere_test_results_PMCMRplus.json", 'r') as f:
#     pcmr_benchmarks = json.load(f)
# with open("C:\\Users\\matth\\OneDrive\\Documents\\jonckheere_test_results_clinfun2.json", 'r') as f:
#     clinfun_benchbarks = json.load(f)
#
#
# def is_close(a, b, tol=1e-4):
#     return abs(a - b) < tol
#
#
# def jonckheere_test_results_PMCMRplus():
#     failed = []
#     for record in pcmr_benchmarks:
#         print(f"Running test case {record['id']}...")
#         x = np.array(record["x"])
#         g = np.array(record["g"])
#         alternative = record["alternative"]
#
#         statistic, p_value = jonckheere_terpstra_test(x, g, alternative=alternative, nperm=5000)
#
#         passed_stat = is_close(statistic, record["statistic"], tol=1e-6)
#         passed_pval = is_close(p_value, record["p_value"], tol=5e-3)
#
#         if passed_stat and passed_pval:
#             print(f"✅ Test {record['id']} passed")
#             failed.append({
#                 'nperm': 5000,
#                 'slope': record['slope'],
#                 'alternative': record['alternative'],
#                 'number': len(record['g']),
#                 'true_p': round(record['p_value'], 3),
#                 'our_p': round(p_value, 3),
#                 'true_stat': record['statistic'],
#                 'our_stat': statistic,
#                 'n': x,
#                 'len(np.unique(x)) != n': len(np.unique(x)) != len(x),
#                 'failed': False
#             })
#         else:
#             print(f"❌ Test {record['id']} failed")
#             print(record['alternative'])
#             print(record['slope'])
#             print(len(record['g']))
#             print(np.unique(np.array(record['g'])))
#             print(f"  Expected statistic: {record['statistic']} vs got {statistic}")
#             print(f"  Expected p-value: {record['p_value']} vs got {p_value}")
#             print('##############################################')
#             failed.append({
#                 'nperm': 5000,
#                 'slope': record['slope'],
#                 'alternative': record['alternative'],
#                 'number': len(record['g']),
#                 'true_p': round(record['p_value'], 3),
#                 'our_p': round(p_value, 3),
#                 'true_stat': record['statistic'],
#                 'our_stat': statistic,
#                 'n': len(x),
#                 'len(np.unique(x)) != n': len(np.unique(x)) != len(x),
#                 'failed': True
#             })
#
#     import pandas as pd
#
#     df = pd.DataFrame(failed)
#     df = df.sort_values(by='failed', ascending=False)
#
#
# benchmark = pcmr_benchmarks
# status = []
# for id_value, record in enumerate(benchmark):
#     x = np.array(record["x"])
#     g = np.array(record["g"])
#     slope = record["slope"]
#     group_size = record["group_size"]
#     group_number = record["group_number"]
#     alternative = record["alt"]
#     ties = record["ties"]
#     more_than_100_obs = record["more_than_100_obs"]
#
#     test_p_value = record["p_value"]
#     test_statistic = record["statistic"]
#     test_significant = record["significant"]
#     test_zstat = record.get("zstat")
#
#     nperm = record.get("nperm")
#     if isinstance(nperm, int):
#         nperm = nperm
#     else:
#         nperm = None
#     continuity = record.get("continuity")
#     if isinstance(continuity, int):
#         continuity = continuity
#     else:
#         continuity = True
#     print(f"Running test case {id_value}...")
#     statistic, p_value, zstat = jonckheere_terpstra_test(x, g,
#                                                          alternative=alternative,
#                                                          nperm=nperm,
#                                                          continuity=continuity
#                                                          )
#     passed_stat = is_close(statistic, test_statistic, tol=1e-6)
#     passed_pval = is_close(p_value, test_p_value, tol=0.01)
#     if test_zstat:
#         passed_zstat = is_close(zstat, test_zstat, tol=0.01)
#     else:
#         passed_zstat = True
#     result = record.copy()
#     result.update({'result_statistic': statistic,
#                    'result_p_value': p_value,
#                    'result_zstat': zstat,
#                    })
#     if passed_stat and passed_pval and passed_zstat:
#         result.update({'failed': False})
#     else:
#         result.update({'failed': True})
#     status.append(result)
# import pandas as pd
#
# df = pd.DataFrame(status)
# df = df.sort_values(by='failed', ascending=False)
# df.to_clipboard()
