# Model review — DVC + CML

**Quality gate:** FAIL — do not merge

| Metric | Base | Pull request | Delta |
|:--|--:|--:|--:|
| accuracy | 0.8333 | 0.7567 | -0.0767 |
| f1 | 0.8344 | 0.7591 | -0.0754 |
| roc_auc | 0.9140 | 0.8167 | -0.0973 |

![ROC curves](./reports/roc_comparison.png)

### Merge policy
- Accuracy >= 0.750
- Accuracy drop <= 0.010
- ROC AUC drop <= 0.010

### Violations
- accuracy dropped more than 0.010
- roc_auc dropped more than 0.010
