# Legacy scripts

Scripts in this folder are retained only for provenance of earlier exploratory
analyses. They are not part of the workflow that produces the results reported
in the final manuscript.

- `auc_inference.py` — bootstrap confidence intervals and a paired sign-flip
  test for mean within-query AUC. Superseded because, with one genuine
  reference per query and a fixed reference-database size, within-query AUC is
  a fixed transformation of the true-match rank; the final manuscript reports
  mean AUC only as a rank-derived summary produced by
  `05_statistics/rank_inference.py`.
