# Table 2 — Pairwise Bootstrap Significance Test (SASContrast vs. SimSiam)

**Requested by:** reviewer, in response to concern that Table 2 gains over
SimSiam/BYOL are not statistically confirmed.
**Method:** paired bootstrap, n = 1000 resamples, resampled at the LOSO-fold
(subject) level — the same unit Table 2's mean F1/Accuracy is already
averaged over.
**Script:** `analysis/table2_bootstrap_significance_test.py` (deterministic,
seeded; rerunning reproduces the numbers below exactly).
**Scope:** this note covers only the SASContrast vs. SimSiam pairs the raw
per-fold predictions were available for (BYOL predictions were not part of
this export and are not covered here).

## Method

For each pretrain→finetune setting, macro-F1 and Accuracy were recomputed
directly from the exported per-fold prediction files
(`finetuned_best_{fold}.csv`, columns `label,pred`) rather than trusted from
the logged summary CSVs. For each of 1000 draws, the set of folds is
resampled with replacement and the mean SASContrast − SimSiam difference is
recomputed; the resulting distribution gives a 95% percentile CI and a
two-sided p-value (p = 2·min(P(diff≤0), P(diff≥0)) under resampling).

This fold-level design was chosen — over a finer-grained instance-level
bootstrap — for two reasons: (1) it directly targets the quantity actually
reported in Table 2 (an unweighted mean over folds/subjects), and (2) a
finer per-instance pairing turned out to be invalid here (see caveat below).

## Data-integrity checks performed first

Before running any test, every fold's recomputed F1/Accuracy was
cross-checked against the corresponding row logged in that run's
`results.csv`:

- **7 of 8 run directories:** exact match on every fold.
- **`Simsiam/simsiam_SWELLDataset_0/finetune/WESADDataset`, fold 0:** the
  on-disk prediction file (F1 = 0.4054) does **not** match the score logged
  for that fold in Table 2 (F1 = 0.5635) — it was silently overwritten by a
  later, unlogged rerun. This fold was **excluded from both arms** of the
  SWELL→WESAD comparison to keep the pairing intact (n = 14 instead of 15).
  This is a provenance issue worth fixing in the training pipeline (append a
  results row on every finetune, or don't allow re-running a single fold
  without regenerating the aggregate log).
- **Caveat on row order:** an instance-level (per-row) paired bootstrap was
  attempted initially but abandoned after finding that prediction files for
  the *same* fold index are not always written in the same row order across
  different training runs (label *multiset* matches, row-by-row order does
  not, confirmed for the SWELL-pretrained→WESAD pair). Since the two models'
  CSVs cannot be assumed row-aligned in general, only fold-level aggregate
  metrics (which don't depend on in-file row order) are reported.

## Results (n = 1000 bootstrap resamples)

| Setting | n folds | SASContrast F1 | SimSiam F1 | ΔF1 | 95% CI (ΔF1) | p (F1) | ΔAcc | 95% CI (ΔAcc) | p (Acc) | Significant (p<0.05) |
|---|---|---|---|---|---|---|---|---|---|---|
| PsychioNet → WESAD | 15 | 0.9073 | 0.8523 | +0.0551 | [−0.010, 0.121] | 0.130 | +0.0401 | [−0.017, 0.097] | 0.148 | **No** |
| PsychioNet → SWELL | 25 | 0.6818 | 0.5969 | +0.0850 | [0.028, 0.141] | 0.000 | +0.0836 | [0.046, 0.123] | 0.000 | Yes |
| WESAD → SWELL | 25 | 0.6686 | 0.6127 | +0.0558 | [0.019, 0.093] | 0.006 | +0.0520 | [0.027, 0.081] | 0.000 | Yes |
| SWELL → WESAD* | 14 | 0.9126 | 0.8295 | +0.0831 | [0.028, 0.149] | 0.002 | +0.0606 | [0.023, 0.101] | 0.000 | Yes |

*fold 0 excluded, see data-integrity note above.

## Verdict

**Partially confirmed — not uniformly.** 3 of the 4 comparisons show a
statistically significant SASContrast advantage at p < 0.05 (both F1 and
Accuracy), with 95% CIs on the mean fold-level difference excluding zero.

The **PsychioNet-pretrained → WESAD** comparison does **not** reach
significance (p ≈ 0.13 for F1, p ≈ 0.15 for Accuracy); its 95% CI
(−0.010 to 0.121 for ΔF1) includes zero. This is not noise in the test
itself — it reflects genuinely high fold-to-fold variance in that setting:
of 15 LOSO folds, 5 have SimSiam matching or beating SASContrast (e.g. fold 0:
0.710 vs. 0.896; fold 6: 0.983 vs. 0.988), while others favor SASContrast by
a wide margin (fold 2: 0.936 vs. 0.641). With only 15 subjects, that
sign-flipping pattern is enough to prevent the mean gain from clearing the
0.05 threshold under resampling, even though the point-estimate gap
(+5.5 F1 points) is the same order of magnitude as the other three settings.

## Recommended framing for the rebuttal

- Report the table above verbatim, with p-values, as the reviewer requested.
- Claim significance where it holds (3/4 settings) rather than a blanket
  claim across all of Table 2.
- For PsychioNet→WESAD, be upfront that the improvement is directionally
  consistent but not statistically significant at n=15 folds, and frame it
  as a power/variance limitation of LOSO evaluation on a 15-subject dataset
  rather than retracting the result. If additional runs/seeds for this one
  setting are feasible before the camera-ready, that would be the fastest
  way to either firm up or soften this specific claim.
- Disclose the fold-0 data provenance issue for SimSiam SWELL→WESAD and the
  fix applied (excluded, n=14), rather than silently patching it — reviewers
  who dig into supplementary code will find file mtimes if asked.
