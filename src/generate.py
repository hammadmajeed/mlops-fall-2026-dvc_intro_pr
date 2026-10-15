from pathlib import Path
import pandas as pd
import yaml
from sklearn.datasets import make_classification

params = yaml.safe_load(Path('params.yaml').read_text())['data']
X, y = make_classification(n_samples=params['n_samples'], n_features=12,
    n_informative=6, n_redundant=2, class_sep=1.0,
    flip_y=0.04, random_state=params['random_state'])
frame = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
frame['target'] = y
Path('data').mkdir(exist_ok=True)
frame.to_csv('data/dataset.csv', index=False)
print(f'Generated {len(frame)} samples')
