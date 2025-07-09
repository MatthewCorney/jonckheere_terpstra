"""Taken from https://pypi.org/project/regressionpack/1.0.5/#files manually due to versioning issues"""
from typing import Any
from scipy.special import factorial
import scipy.stats


def computeJ(groups: list[list[float]]):
    # Computes the Jonckheere statistic J
    J = 0
    for i, groupA in enumerate(groups[:-1]):
        for groupB in groups[i + 1:]:
            for sampleA in groupA:
                for sampleB in groupB:
                    if sampleA < sampleB:
                        J += 1

    return J


def combinationsWithComplement(iterable, r) -> list[Any]:
    """Pretty much the same thing as itertools.combinations, except
    it will also return the complement. For example, let's say you do
    combinationsWithComplement([1,2,3,4],2)
    the generator will yield (combination, complement) pairs, such as:
    ((1, 2), (3, 4))
    ((1, 3), (2, 4))
    ((1, 4), (2, 3))
    ((2, 3), (1, 4))
    ((2, 4), (1, 3))
    ((3, 4), (1, 2))
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices), tuple(value for i, value in enumerate(pool) if i not in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices), tuple(value for i, value in enumerate(pool) if i not in indices)


def combinationsAsPartitions(pool: list[Any], nb: list[int]) -> list[Any]:
    """Starting from a flat list, and a list of how long each set must be,
    this will generate every possible arrangement of sets. It will still be flat though,
    and you will have to re-create the groups.
    """
    assert len(pool) == sum(
        nb), "The length of the pool must equal the sum of the lengths of the individual partitions. "
    return _combinationsAsPartitions(pool, nb)


def _combinationsAsPartitions(pool: list[Any], nb: list[int]) -> list[Any]:
    if len(nb) == 1:
        yield pool
        return

    for group, others in combinationsWithComplement(pool, nb[0]):

        for othergroup in _combinationsAsPartitions(others, nb[1:]):
            yield group + othergroup


def createSizedGroups(pool: list[Any], nb: list[int]) -> list[tuple[Any]]:
    """Takes in a list of numbers, and returns a list of lists of the sizes provided in nb.
    """
    assert len(pool) == sum(
        nb), "The length of the pool must equal the sum of the lengths of the individual partitions. "
    ans = list()
    currentPosition = 0
    for n in nb:
        ans.append(tuple(pool[currentPosition:currentPosition + n]))
        currentPosition += n
    return ans


def generateSizedPartitions(pool: list[Any], nb: list[int]) -> list[tuple[Any]]:
    """Starting from a flat list, and a list of how long each set must be,
    this will generate every possible arrangement of sets.
    """
    for rawPartition in combinationsAsPartitions(pool, nb):
        yield createSizedGroups(rawPartition, nb)


class JonckheereTerpstra:
    Groups: list[list[float]]
    Nb: list[int]
    N: int
    J: int
    NumberOfComparisons: int
    S: int
    RankCorrelation: float
    Esperance: float
    Variance: float
    Z: float

    NumberOfPossibleValues: int
    Degenerescency: int
    NumberOfValidPermutations: int

    def __init__(self, groups: list[list[float]]):
        self.Groups = groups

        # Preparation
        self.Nb = [len(group) for group in self.Groups]
        self.N = sum(self.Nb)
        self._CountComparisons()

        # The Statistics
        self.J = computeJ(self.Groups)
        self.S = 2 * self.J - self.NumberOfComparisons

        # Values from the statistic
        self.RankCorrelation = self.S / self.NumberOfComparisons

        self.Z = None
        self.NumberOfValidPermutations = None

    def _CountComparisons(self):
        self.NumberOfComparisons = 0
        for i, ni in enumerate(self.Nb):
            for nj in self.Nb[i + 1:]:
                self.NumberOfComparisons += ni * nj

    def ComputeApproximateProbability(self) -> float:

        if self.Z is None:
            self.Esperance = (self.N ** 2 - sum([n ** 2 for n in self.Nb])) / 4
            self.Variance = (self.N ** 2 * (2 * self.N + 3) - sum([(2 * nj + 3) * nj ** 2 for nj in self.Nb])) / 72
            self.Z = (self.J - 0.5 - self.Esperance) / self.Variance ** 0.5

        return 1 - scipy.stats.norm.cdf(self.Z)

    def ComputeExactProbability(self):

        if self.NumberOfValidPermutations is None:

            self.NumberOfPossibleValues = factorial(self.N)
            self.Degenerescency = 1
            for n in self.Nb:
                self.Degenerescency *= factorial(n)

            self.NumberOfPossibleValues /= self.Degenerescency

            # Do the permutations of the groups, and count how many J's are
            # greater or equal to J0 (the actual one we have now)

            flatSet = list()
            for group in self.Groups:
                flatSet += group

            self.NumberOfValidPermutations = 0
            for permutation in generateSizedPartitions(flatSet, self.Nb):

                if computeJ(permutation) >= self.J:
                    self.NumberOfValidPermutations += 1

        return self.NumberOfValidPermutations / self.NumberOfPossibleValues


def main():
    """This function should print the results below. Please note that it
    will take a few seconds (around 10) to compute the exact probability
    as we have to compute the Jonckheere statistic for all the possible
    permutations of the values.

    Reproducing example from https://doi.org/10.2466%2Fpms.1997.85.1.107
    W0 = 59
    mu_W = 37.5
    VAR_W = 89.58333333333333
    Z = 2.218736283940572
    Approximated probability: 0.013252335504576052
    Exact probability: 0.011964226249940535

    """

    print("Reproducing example from https://doi.org/10.2466%2Fpms.1997.85.1.107")

    groups = [
        [99, 114, 116, 127, 146],
        [111, 125, 143, 148, 157],
        [133, 139, 149, 160, 184]
    ]

    jt = JonckheereTerpstra(groups)

    approximateProbability = jt.ComputeApproximateProbability()
    exactProbability = jt.ComputeExactProbability()
    print(f"W0 = {jt.J}")
    print(f"mu_W = {jt.Esperance}")
    print(f"VAR_W = {jt.Variance}")
    print(f"Z = {jt.Z}")
    print(f"Approximated probability: {approximateProbability}")
    print(f"Exact probability: {exactProbability}")


if __name__ == "__main__":
    main()