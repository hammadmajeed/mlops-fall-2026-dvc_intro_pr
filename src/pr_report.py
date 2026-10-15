"""Compare PR vs base Git revision and create CML report plus PNG chart."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def git_file(ref, name):
    result = subprocess.run(['git', 'show', f'{ref}:{name}'],
                            capture_output=True, text=True, check=True)
    return result.stdout


def decision(base, candidate, policy):
    issues = []
    if candidate['accuracy'] < policy['min_accuracy']:
        issues.append('Candidate accuracy below absolute minimum')
    for metric, tolerance in [('accuracy', 'max_accuracy_drop'), ('roc_auc', 'max_auc_drop')]:
        if candidate[metric] < base[metric] - policy[tolerance]:
            issues.append(f'{metric} dropped more than {policy[tolerance]:.3f}')
    return issues


def create(base_ref):
    base = json.loads(git_file(base_ref, 'metrics.json'))
    candidate = json.loads(Path('metrics.json').read_text())
    base_roc = pd.read_csv(io.StringIO(git_file(base_ref, 'plots/roc.csv')))
    new_roc = pd.read_csv('plots/roc.csv')
    policy = yaml.safe_load(Path('params.yaml').read_text())['policy']
    issues = decision(base, candidate, policy)
    Path('reports').mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(base_roc.fpr, base_roc.tpr, label=f"Base (AUC {base['roc_auc']:.3f})", lw=2)
    ax.plot(new_roc.fpr, new_roc.tpr, label=f"PR (AUC {candidate['roc_auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random classifier')
    ax.set(xlabel='False positive rate', ylabel='True positive rate',
           title='ROC curve: base branch vs pull request', xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc='lower right')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('reports/roc_comparison.png', dpi=160)
    plt.close(fig)
    lines = ['# Model review — DVC + CML', '',
             '**Quality gate:** ' + ('FAIL — do not merge' if issues else 'PASS — eligible for human review'),
             '', '| Metric | Base | Pull request | Delta |',
             '|:--|--:|--:|--:|']
    for metric in ('accuracy', 'f1', 'roc_auc'):
        b, c = base[metric], candidate[metric]
        lines.append(f'| {metric} | {b:.4f} | {c:.4f} | {c-b:+.4f} |')
    lines += ['', '![ROC curves](./reports/roc_comparison.png)', '',
              '### Merge policy',
              f"- Accuracy >= {policy['min_accuracy']:.3f}",
              f"- Accuracy drop <= {policy['max_accuracy_drop']:.3f}",
              f"- ROC AUC drop <= {policy['max_auc_drop']:.3f}", '']
    if issues:
        lines += ['### Violations'] + [f'- {issue}' for issue in issues]
    else:
        lines += ['No metric-gate violations. Review data quality, fairness, and deployment constraints before merging.']
    Path('report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:9]))
    return bool(issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ref', default='origin/main')
    args = parser.parse_args()
    raise SystemExit(1 if create(args.base_ref) else 0)
