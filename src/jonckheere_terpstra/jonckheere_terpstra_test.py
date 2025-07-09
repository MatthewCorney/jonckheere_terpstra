import warnings

import numpy as np
import pandas as pd

from scipy.stats import page_trend_test
from scipy.stats import norm

from typing import Literal
from typing import Tuple
from typing import Optional
from typing import Union

from collections import Counter
from more_itertools import distinct_permutations

_ValidAlternatives = Literal['increasing', 'decreasing', 'two_sided']
_ValidMethods = Literal['permutation', 'approximate', 'exact']


def _compute_jt_statistic(x: np.ndarray[np.number], g: np.ndarray[int]) -> int:
    """
    Compute the Jonckheere–Terpstra U statistic directly
    from global ranks of x, given ordered group labels g.

    :param x: Ordered numeric response variable
    :param g: Group labels array

    :return int: The JT test statistic
    """

    _, counts = np.unique(g, return_counts=True)
    cg: np.ndarray = np.concatenate(([0], np.cumsum(counts)))

    u = 0
    # for each ordered pair of groups (i<j)
    for i in range(len(counts) - 1):
        xi = x[cg[i]: cg[i + 1]]
        for j in range(i + 1, len(counts)):
            xj = x[cg[j]: cg[j + 1]]
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


def _jt_approximate_pvalue(jt_stat: float,
                           jtmean: float,
                           jtvar: float,
                           alternative: _ValidAlternatives,
                           continuity: bool) -> Tuple[float, float]:
    """
    Compute the approximate (asymptotic) p-value and z-statistic for the Jonckheere–Terpstra test.

    Uses a normal approximation under the null hypothesis to compute a z-score from the observed
    JT statistic, with optional continuity correction.

    :param jt_stat: Observed Jonckheere–Terpstra statistic
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
    adjustment = delta * np.sign(jt_stat - jtmean)
    zstat = (jt_stat - jtmean - adjustment) / np.sqrt(jtvar)
    if alternative == "two_sided":
        pval = 2 * norm.sf(abs(zstat))
    elif alternative == "increasing":
        pval = norm.sf(zstat)
    elif alternative == "decreasing":
        pval = norm.cdf(zstat)
    else:
        raise ValueError("Invalid alternative")
    return float(pval.item()), float(zstat.item())


def _jt_permutation_pvalue(x: np.ndarray[np.number],
                           gsize: np.ndarray[int],
                           alternative: str,
                           nperm: int) -> Tuple[float, float]:
    """
    Compute a permutation-based p-value for the Jonckheere–Terpstra test.

    This function generates a null distribution of the JT statistic by resampling the data `nperm` times.
    The first value in the permutation distribution corresponds to the original, unpermuted data.

    :param x: Data values (must be numeric)
    :param gsize: Sizes of each group, in increasing order
    :param alternative: One of "two_sided", "increasing", or "decreasing"
    :param nperm: Number of permutations to perform

    :return: Tuple containing:
         - p-value (float)
         - z-statistic (float)
    """
    ng = len(gsize)
    perm_jt_stat: np.ndarray = np.empty(nperm)
    x_perm = x.copy()

    # compute statistic on original and permuted data sequentially
    for i in range(nperm):
        x_perm = np.random.permutation(x_perm)
        g_indices = np.repeat(np.arange(ng), gsize)  # Recreate group indices
        jt_stat = _compute_jt_statistic(x_perm, g_indices)
        perm_jt_stat[i] = jt_stat
    jtr0 = perm_jt_stat[0]
    null_stats = perm_jt_stat[1:]
    null_mean = np.mean(null_stats)
    null_std = np.std(null_stats, ddof=1)
    zstat = (jtr0 - null_mean) / null_std

    ipval = float(np.mean(perm_jt_stat <= jtr0))
    dpval = float(np.mean(perm_jt_stat >= jtr0))

    if alternative == "two_sided":
        return 2 * min([ipval, dpval, 0.5]), float(zstat.item())
    elif alternative == "increasing":
        return ipval, zstat
    elif alternative == "decreasing":
        return dpval, zstat
    else:
        raise ValueError("Invalid alternative")


def _compute_jt_statistic_vectorized(ranks: np.ndarray, groups: np.ndarray) -> int:
    """
    Vectorized computation of the Jonckheere–Terpstra U statistic.
    :param ranks: 1-D array of ordinal ranks
    :param groups: 1-D array of group labels (0,1,2,...)
    """
    order = np.argsort(groups)
    r = ranks[order]
    g = groups[order]
    comp = np.less.outer(r, r).astype(int)
    gcomp = np.less.outer(g, g).astype(int)
    return int((comp * gcomp).sum())


def _jt_exact_pvalue(ranks: np.ndarray,
                     gsize: np.ndarray,
                     jt_stat: int,
                     alternative: str) -> Tuple[float, None]:
    """
    Optimized exact p-value calculation via enumeration of unique permutations.
    Uses Counter over distinct permutations to avoid redundant computations when ties are present.

    :param ranks: 1-D array of ordinal ranks (ties allowed)
    :param gsize: 1-D array of group sizes
    :param jt_stat: Observed JT statistic
    :param alternative: 'increasing', 'decreasing', or 'two_sided'
    :return: (p-value, None)
    """
    # Precompute group labels
    groups = np.repeat(np.arange(len(gsize)), gsize)

    # Count JT statistic frequencies over unique permutations
    stat_counts = Counter(
        _compute_jt_statistic_vectorized(np.array(perm), groups)
        for perm in distinct_permutations(ranks)
    )
    total = sum(stat_counts.values())

    # Compute p-value based on alternative
    if alternative == 'increasing':
        # sum frequencies >= observed
        pval = sum(count for stat, count in stat_counts.items() if stat >= jt_stat) / total
    elif alternative == 'decreasing':
        # sum frequencies <= observed
        pval = sum(count for stat, count in stat_counts.items() if stat <= jt_stat) / total
    elif alternative == 'two_sided':
        p_inc = sum(count for stat, count in stat_counts.items() if stat >= jt_stat) / total
        p_dec = sum(count for stat, count in stat_counts.items() if stat <= jt_stat) / total
        pval = 2 * min(p_inc, p_dec)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    return float(pval), None


def jonckheere_terpstra_test(x: Union[np.ndarray, list],
                             g: Union[np.ndarray, list],
                             alternative: _ValidAlternatives = "two_sided",
                             method: _ValidMethods = 'approximate',
                             nperm: Optional[int] = None,
                             continuity: bool = True) -> tuple[int, float, Optional[float]]:
    """
    Perform the Jonckheere–Terpstra test for ordered differences among multiple groups.

    The test evaluates whether there is a trend (increasing, decreasing, or two-sided) in the distribution
    of a numeric variable `x` across ordered groups `g`.

    :param method: How to calculate the P-Value, defaults to approximate
    :param x: Numeric observations variable
    :param g: Group labels (must be orderable; numeric or ordinal)
    :param alternative: Direction of the trend to test. One of:
                        - "two_sided": test for any ordered difference
                        - "increasing": test for an increasing trend across groups
                        - "decreasing": test for a decreasing trend across groups
    :param nperm: If provided, compute p-value using `nperm` permutations instead of asymptotic or exact method
    :param continuity: if continuity correction should be used in the approximate case

    :return: Tuple containing:
         - JT statistic (int)
         - p-value (float)
         - z-statistic (float)
    """
    if alternative not in ['increasing', 'decreasing', 'two_sided']:
        raise ValueError(f'Alternative must be one of {_ValidAlternatives} {alternative=} passed')
    if method not in ['permutation', 'approximate', 'exact']:
        raise ValueError(f'Method must be one of {_ValidMethods} {method=} passed')
    x: np.ndarray[np.number] = np.asarray(x)
    g: np.ndarray[int] = np.asarray(g)
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("x must be numeric")
    if len(x) != len(g):
        raise ValueError("x and g must be the same length")
    if len(np.unique(g)) < 3:
        raise ValueError("Jonckheere–Terpstra test requires at least 3 ordered groups")
    if len(np.unique(x)) < len(x) and not (method == 'permutation'):
        warnings.warn("Permutation should be used if ties are present")

    finite = np.isfinite(x) & np.isfinite(g)
    x = x[finite]
    g = g[finite]

    order = np.argsort(g)
    x = x[order]
    g = g[order]

    _, counts = np.unique(g, return_counts=True)
    gsize: np.ndarray[int] = counts
    ng: int = len(gsize)
    cgsize: np.ndarray[int] = np.concatenate(([0], np.cumsum(gsize)))
    n: int = len(x)

    # compute mean and  var
    jtmean, jtvar = _calculate_mean_variance(n, ng, gsize, cgsize)

    # compute jt statistic
    jt_stat = _compute_jt_statistic(x, g)
    if method == 'permutation':
        if nperm is None:
            raise ValueError("You must specify nperm for permutation method.")
        pval, zstat = _jt_permutation_pvalue(x=x,
                                             gsize=gsize,
                                             alternative=alternative,
                                             nperm=nperm
                                             )
    elif method == 'exact':
        if n > 10:  # Adjust threshold as needed
            warnings.warn(f"Exact method with n={n} may be very slow. Consider 'approximate' or 'permutation' methods.")
        ranks = np.argsort(np.argsort(x))
        pval, zstat = _jt_exact_pvalue(
            ranks=ranks, gsize=gsize,
            jt_stat=jt_stat, alternative=alternative
        )
    else:
        pval, zstat = _jt_approximate_pvalue(jtmean=jtmean,
                                             jtvar=jtvar,
                                             jt_stat=jt_stat,
                                             alternative=alternative,
                                             continuity=continuity
                                             )
    return jt_stat, pval, zstat


def pages_l_test(x: Union[np.ndarray, list],
                 g: Union[np.ndarray, list],
                 s: Union[np.ndarray, list],
                 alternative: _ValidAlternatives = "two_sided", ) -> Tuple[float, float]:
    """
    Wrapper for the scipy Page's L test for ordered alternatives across treatments.

    Page's trend test is a non-parametric test used to detect a monotonic trend
    (increasing or decreasing) across multiple treatments, with repeated measures
    for each subject.

    :param x: Numeric observations variable
    :param g: Group labels (must be orderable; numeric or ordinal)
    :param s: Subject identifiers
    :param alternative: Direction of the alternative hypothesis:
                        - 'increasing' (default): test for monotonic increase.
                        - 'decreasing': test for monotonic decrease.
                        - 'two_sided': tests both directions by doubling the smaller p-value.

    :return: Tuple containing:
             - Pages L statistic (float)
             - p-value (float)
    """
    x: np.ndarray[np.number] = np.asarray(x)
    g: np.ndarray[int] = np.asarray(g)
    s: np.ndarray = np.asarray(s)
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("x must be numeric")
    if len(x) != len(g):
        raise ValueError("x and g must be the same length")
    if len(np.unique(g)) < 3:
        raise ValueError("Pages L test requires at least 3 ordered groups")
    if alternative not in ['increasing', 'decreasing', 'two_sided']:
        raise ValueError(f'Alternative must be one of {_ValidAlternatives} {alternative=} passed')

    df = pd.DataFrame({'s': s, 'g': g, 'x': x})
    df_pivot = df.pivot(index='s', columns='g', values='x')
    df_pivot = df_pivot[list(sorted((set(g))))]

    data = df_pivot.to_numpy()
    if alternative == 'increasing':
        result = page_trend_test(data)
        statistic, pval = result.statistic, result.pvalue
    elif alternative == 'decreasing':
        result = page_trend_test(data[:, ::-1])
        statistic, pval = result.statistic, result.pvalue
    elif alternative == 'two_sided':
        iresult = page_trend_test(data)
        dresult = page_trend_test(data[:, ::-1])
        if iresult.pvalue < dresult.pvalue:
            pval = min(2 * iresult.pvalue, 1)
            statistic = iresult.statistic
        else:
            pval = min(2 * dresult.pvalue, 1)
            statistic = dresult.statistic
    else:
        raise ValueError("Invalid alternative")
    return float(pval), float(statistic)
