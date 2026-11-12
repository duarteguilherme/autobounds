# Data for the JASA paper examples

These nine CSV files are byte-for-byte copies of the corresponding files in
`replication_files/data/`. They contain the simulated observed probabilities
used by the original replication scripts, not newly generated datasets.

Source: Duarte, Finkelstein, Knox, Mummolo, and Shpitser (2024),
[*An Automated Approach to Causal Inference in Discrete Settings*](https://doi.org/10.1080/01621459.2023.2216909),
JASA 119(547), 1778–1793. The article was first published online in 2023.

| File | Meaning |
| --- | --- |
| `iv.csv` | Joint P(Z,X,Y), Section 7.1 |
| `selection_obsqty.csv` | Joint P(X,Y,S=1), Figure 7(a) |
| `measurement_error.csv` | Joint P(X,S), Figure 7(b); S is the measured proxy |
| `manski1_1.csv` | Conditional P(X,Y\|R=1), Figure 7(c) |
| `manski1_2.csv` | Joint P(X,R), Figure 7(c) |
| `manski2_obs_x_rx1.csv` | Joint P(X,A=1), Figure 7(d) |
| `manski2_obs_rx_ry.csv` | Joint P(A,B), Figure 7(d) |
| `manski2_obs_x_y_rx1_ry1.csv` | Joint P(X,Y,A=1,B=1), Figure 7(d) |
| `manski2_obs_y_ry1.csv` | Joint P(Y,B=1), Figure 7(d) |

Here A and B indicate whether X and Y are observed. Tables containing only
selected cells generally do **not** sum to one. The runner preserves those
probabilities; renormalizing them would change the empirical evidence.
The joint-missingness runner derives P(X) from P(X,A=1)/P(A=1), using the graph's
independence of A and X. It does not modify any input file.

The archived CSVs have finite decimal precision. Published endpoints are rounded
as well; numerical comparisons therefore allow appropriate tolerances.
