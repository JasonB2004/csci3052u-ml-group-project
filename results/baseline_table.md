Validation results, official split (seed 42). Mean/std are over split seeds 0-9. Test set not used.

| model | train_macro_f1 | valid_macro_f1 | valid_macro_f1_mean_10_seeds | valid_macro_f1_std_10_seeds | valid_accuracy | valid_recall_Dropout | valid_recall_Enrolled | valid_recall_Graduate |
|---|---|---|---|---|---|---|---|---|
| Majority class | 0.222 | 0.222 | 0.222 | 0.0 | 0.5 | 0.0 | 0.0 | 1.0 |
| Stratified random | 0.329 | 0.313 | 0.343 | 0.012 | 0.373 | 0.319 | 0.126 | 0.497 |
| Softmax regression (NumPy GD) | 0.705 | 0.661 | 0.667 | 0.02 | 0.742 | 0.7 | 0.353 | 0.91 |
| sklearn LogisticRegression (check) | 0.719 | 0.651 | - | - | 0.724 | 0.695 | 0.37 | 0.87 |
| Softmax regression + 2nd-sem features | 0.741 | 0.695 | - | - | 0.765 | 0.737 | 0.412 | 0.91 |
