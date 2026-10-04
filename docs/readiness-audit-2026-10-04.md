# Controlled Experiment Readiness Audit

The read-only audit for `configs/controlled_stage_holdout_v3.yaml` passed all
diagnostic checks:

- config and manifest both retain `formal_result=false`;
- seeds `5..9` and 200 episodes per role match;
- train, support, query, qualification, and SPT-validation episode IDs are
  disjoint for every seed;
- qualification, Knowledge, SPT, and query-efficiency fields are configured;
- the recorded Git commit is `c5e15c4` and the required Python executable is
  the `kecrl` environment.

The machine-readable report is
`results/holdout_prior_v3/readiness_audit.json`. This grants readiness for a
diagnostic run only. The formal-result gate remains blocked until the real
backend scope and statistical procedure are frozen.
