from pathlib import Path
import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

p = yaml.safe_load(Path('params.yaml').read_text())['train']
df = pd.read_csv('data/dataset.csv')
X = df.drop(columns=['target'])
y = df['target']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=p['test_size'],
    stratify=y, random_state=p['random_state'])
model = RandomForestClassifier(n_estimators=p['n_estimators'],
    max_depth=p['max_depth'], random_state=p['random_state'], n_jobs=1)
model.fit(Xtr, ytr)
Path('models').mkdir(exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
Path('data/test.csv').write_text(Xte.assign(target=yte).to_csv(index=False))
