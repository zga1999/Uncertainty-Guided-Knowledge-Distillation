"""Summarize MMEngine JSON-lines logs without loading model checkpoints."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


TRAIN_FIELDS = [
    'epoch', 'iter', 'step', 'lr', 'loss', 'loss_ori', 'loss_uatm',
    'loss_ugsa', 'time', 'data_time', 'memory',
]
LOSS_FIELDS = ['loss', 'loss_ori', 'loss_uatm', 'loss_ugsa']


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summarize(scalars, output, expected_epochs=None):
    training, validation = [], []
    for number, line in enumerate(scalars.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if 'accuracy/top1' in row:
            epoch = int(row['step'])
            if row['step'] != epoch:
                raise ValueError(f'Nonintegral validation step at line {number}')
            value = float(row['accuracy/top1'])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'Invalid accuracy at line {number}')
            validation.append(dict(epoch=epoch, top1_percent=value,
                                   data_time=row.get('data_time'), time=row.get('time')))
        elif 'epoch' in row and 'loss' in row:
            training.append(row)
    if not validation:
        raise ValueError('No validation accuracy found')
    validation.sort(key=lambda row: row['epoch'])
    epochs = [row['epoch'] for row in validation]
    if len(epochs) != len(set(epochs)):
        raise ValueError('Duplicate validation epochs require explicit run separation')
    if expected_epochs is not None and epochs != list(range(1, expected_epochs + 1)):
        raise ValueError('Validation epochs are incomplete or unexpected')
    best_value = max(row['top1_percent'] for row in validation)
    best_epochs = [row['epoch'] for row in validation if row['top1_percent'] == best_value]
    grouped = defaultdict(list)
    for row in training:
        grouped[int(row['epoch'])].append(row)
    sampled = []
    for epoch, rows in sorted(grouped.items()):
        item = {'epoch': epoch, 'logged_windows': len(rows)}
        for key in LOSS_FIELDS:
            values = [float(row[key]) for row in rows if key in row]
            if values:
                item[key] = mean(values)
        sampled.append(item)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'validation.csv', validation, ['epoch', 'top1_percent', 'data_time', 'time'])
    write_csv(output / 'training.csv', training, TRAIN_FIELDS)
    write_csv(output / 'sampled_training_by_epoch.csv', sampled,
              ['epoch', 'logged_windows'] + LOSS_FIELDS)
    result = {
        'metric': 'accuracy/top1', 'unit': 'percent',
        'best_top1_percent': best_value, 'best_epochs': best_epochs,
        'final_top1_percent': validation[-1]['top1_percent'],
        'final_epoch': validation[-1]['epoch'],
        'validation_records': len(validation), 'training_records': len(training),
        'validation_epoch_range': [epochs[0], epochs[-1]],
        'training_aggregation': 'Unweighted mean of logged windows within each epoch; not a full-epoch loss.',
    }
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result, validation, sampled


def plot(validation, sampled, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    x = [row['epoch'] for row in validation]
    y = [row['top1_percent'] for row in validation]
    best = max(range(len(y)), key=y.__getitem__)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout='constrained')
    ax.plot(x, y, color='#147d92', linewidth=1.5)
    ax.scatter([x[best]], [y[best]], color='#b93f56', s=30, zorder=3)
    ax.annotate(f'{y[best]:.2f}% at epoch {x[best]}', xy=(x[best], y[best]),
                xytext=(-12, -28), textcoords='offset points', ha='right', color='#963349')
    for milestone in [150, 180, 210]:
        ax.axvline(milestone, color='#666666', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set(xlabel='Epoch', ylabel='Top-1 accuracy (%)', xlim=(1, max(x)),
           title='CIFAR-100 | ResNet32x4 to ResNet8x4 | UATM + UGSA')
    ax.grid(axis='y', alpha=0.18)
    for suffix in ['png', 'pdf']:
        fig.savefig(destination / f'validation_top1.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), layout='constrained')
    colors = ['#37474f', '#147d92', '#b93f56', '#688937']
    labels = ['Total loss', 'ORI loss', 'UATM loss', 'UGSA loss']
    for ax, key, color, label in zip(axes.flat, LOSS_FIELDS, colors, labels):
        ax.plot([r['epoch'] for r in sampled], [r[key] for r in sampled],
                color=color, linewidth=1.4)
        ax.set(title=label, xlabel='Epoch', ylabel='Mean of logged windows', xlim=(1, max(x)))
        ax.grid(axis='y', alpha=0.18)
    fig.suptitle('Training losses | sampled logger windows, not full-epoch averages', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(destination / f'training_losses.{suffix}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalars', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-epochs', type=int)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    metrics, validation, sampled = summarize(args.scalars, args.output, args.expected_epochs)
    if args.figures:
        plot(validation, sampled, args.figures)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
