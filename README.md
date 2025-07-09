## Introduction

A lightweight implementation of the Jonckheere–Terpstra test for detecting ordered differences among three or more 
independent groups. This library provides both an asymptotic (normal approximation) and a permutation-based approach to 
compute the p-value and z-statistic for increasing, decreasing, or two-sided trends.

For comparison a wrapper for scipy's Page L test is also included.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.page_trend_test.html

## Dependencies
The function requires the following Python packages:
- numpy
- scipy
- pandas

## Usage
The package can be installed from git with the following command

```
!pip install git+https://github.com/MatthewCorney/jonckheere_terpstra.git
```


## Examples
For the jonckheere_terpstra_test
```python
from jonckheere_terpstra_test import jonckheere_terpstra_test

x = [0.7612, 1.5791, -0.5479, 1.4819, 0.5152, 1.2179, 1.1001, 0.9002, -0.8638, 0.0892, 0.9822, -0.5948, -0.1841, 0.9397, 1.1269, 0.8225, -0.5271, -0.7411, 1.8714, 0.6166, 0.6265, 0.4172, -0.6563, 1.1501, 0.3209, 0.3553, 1.4714, 1.3598, 1.9302, 0.0619]
g = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

jtrsum, pval, zstat = jonckheere_terpstra_test(x=x,
                                             g=g,
                                             continuity=True,
                                             alternative='two_sided'
                                             )
print(f'{jtrsum=}')
print(f'{pval=}')
print(f'{zstat=}')
```
```text
jtrsum=152
pval=0.954
zstat=np.float64(-0.07548688962374275)
```
For the pages_l_test
```python
from jonckheere_terpstra_test import pages_l_test
x = [1, 20, 100, 2, 30, 120, 4, 12, 200]
g = [1, 2, 3, 1, 2, 3, 1, 2, 3]
s = ['S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S3', 'S3', 'S3']

pval, lstat = pages_l_test(x=x,
                         g=g,
                         s=s,
                         alternative='two_sided'
                         )
print(f'{lstat=}')
print(f'{pval=}')
```
```text
lstat = 42.0
pval = 0.009259259259259259
```