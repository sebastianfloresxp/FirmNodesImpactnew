# Chapter 2 Table CLI Cheat Sheet

All commands should be run from the repository root so the default relative paths resolve. Override the
`--meta-*` or `--scores-*` arguments if you need to point at alternate parquet bundles.

| Table | Command | Output fragment |
| --- | --- | --- |
| Tab 2.1 Global Scorecards | `python src/analysis/chapter2/tab_01_global_scorecard.py --output tables/chapter2` | `tables/chapter2/tab_01_global_scorecard_{test,val}.tex` |
| Tab 2.2 Structural & Generalization (Test) | `python src/analysis/chapter2/tab_02_structural_generalization.py --output tables/chapter2/tab_02_structural_generalization.tex` | `tables/chapter2/tab_02_structural_generalization.tex` |
| Tab 2.3 Temporal Horizons | `python src/analysis/chapter2/tab_03_temporal_horizons.py --output tables/chapter2/tab_03_temporal_horizons.tex` | `tables/chapter2/tab_03_temporal_horizons.tex` |
| Appendix A.2 Seed Robustness | `python src/analysis/chapter2/tab_04_seed_robustness.py --output tables/chapter2/tab_04_seed_robustness.tex` | `tables/chapter2/tab_04_seed_robustness.tex` |
| Appendix A.3a Structural Slices (Test) | `python src/analysis/chapter2/tab_05_structural_slices_test.py --output tables/chapter2/tab_05_structural_slices_test.tex` | `tables/chapter2/tab_05_structural_slices_test.tex` |
| Appendix A.3b Structural Slices (Val) | `python src/analysis/chapter2/tab_06_structural_slices_val.py --output tables/chapter2/tab_06_structural_slices_val.tex` | `tables/chapter2/tab_06_structural_slices_val.tex` |
| Appendix A.4 Temporal Horizons | `python src/analysis/chapter2/tab_07_temporal_full.py --output tables/chapter2/tab_07_temporal_horizons_full.tex` | `tables/chapter2/tab_07_temporal_horizons_full.tex` |

Each CLI shares the same core arguments:

```
--meta-val      Validation meta dataset parquet (default: results/ensemble/meta_dataset_v4/meta_inputs_val.parquet)
--meta-test     Test meta dataset parquet (default: results/ensemble/meta_dataset_v4/meta_inputs_test.parquet)
--scores-val    Validation ensemble scores parquet (default: results/ensemble/meta_ranker/meta_ranker_v4/scores_val.parquet)
--scores-test   Test ensemble scores parquet (default: results/ensemble/meta_ranker/meta_ranker_v4/scores_test.parquet)
```

