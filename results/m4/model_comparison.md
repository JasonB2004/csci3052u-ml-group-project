Validation results, official split (seed 42). Mean/std over split seeds 0-9. Test set not used.

| model | train_macro_f1 | valid_macro_f1 | valid_macro_f1_mean_10_seeds | valid_macro_f1_std_10_seeds | valid_accuracy | valid_recall_Dropout | valid_recall_Enrolled | valid_recall_Graduate | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|
| Majority class | 0.222 | 0.222 | 0.222 | 0.0 | 0.5 | 0.0 | 0.0 | 1.0 | - |
| Stratified random | 0.329 | 0.313 | 0.343 | 0.012 | 0.373 | 0.319 | 0.126 | 0.497 | - |
| Softmax regression (M3 baseline) | 0.705 | 0.661 | 0.667 | 0.02 | 0.742 | 0.7 | 0.353 | 0.91 | 3.557 |
| kNN | 0.671 | 0.554 | 0.564 | 0.023 | 0.645 | 0.535 | 0.244 | 0.858 | 0.002 |
| Decision tree | 0.748 | 0.634 | 0.624 | 0.018 | 0.694 | 0.657 | 0.429 | 0.813 | 0.019 |
