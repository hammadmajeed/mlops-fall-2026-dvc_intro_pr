from src.pr_report import decision

POLICY = {'min_accuracy': .75, 'max_accuracy_drop': .01, 'max_auc_drop': .01}

def test_good_model_passes():
    assert decision({'accuracy': .85, 'roc_auc': .90},
                    {'accuracy': .86, 'roc_auc': .91}, POLICY) == []

def test_regression_fails():
    assert decision({'accuracy': .85, 'roc_auc': .90},
                    {'accuracy': .81, 'roc_auc': .85}, POLICY)
