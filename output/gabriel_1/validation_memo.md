# Sales/Marketing GPT Measurement Validation Memo

- Run UTC: 2026-02-26T05:53:01.097535+00:00
- Model: gpt-5.2 (reasoning=medium)
- Prompt selected after one controlled refinement: v2
- Input rows: 16999
- Unique titles scored: 16999

## Pilot and Calibration
- Pilot sample size: 250
- Average absolute score shift (v1->v2): 3.22
- Large shifts (>=25 points): 2

## Parse/Completion
- Parse success is 100% for all completed batches because each batch only checkpoints after strict parse validation.

## Benchmark Sanity Checks
- Pass rate: 1.000

## Validation (Independent QA Coder)
- Overall N: 398
- Spearman(score, manual_proxy): 0.8789
- Weighted kappa (quadratic) on bins: 0.8402

### Split-Level Metrics
- holdout: n=200, spearman=0.7283, weighted_kappa=0.7940
- stratified_qa: n=198, spearman=0.8459, weighted_kappa=0.7810

## Notes
- Prompt tuning was restricted to one controlled refinement from v1 to v2.
- Holdout titles were isolated before pilot prompt refinement.
- Raw API responses are retained in JSONL artifacts for auditability.
- Independent QA coder labels are model-assisted proxy labels, not human-lab-coded annotations.
