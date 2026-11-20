"""Patrones por distribución y tamaño (solo necesita las etiquetas y los cachés).
    python -m label_generation.pattern_analysis [--tag _stub]
Escribe tablas en data/labeled/ y figuras en report/."""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from label_generation.run_labels import OUT_DIR, PRIORITY as CLASSES

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='')
tag = ap.parse_args().tag
FIG = 'report'
DISTS = ['uniform', 'clustered', 'border']

df = pd.concat([pd.read_csv(f'{OUT_DIR}/train_labeled{tag}.csv'),
                pd.read_csv(f'{OUT_DIR}/test_labeled{tag}.csv')], ignore_index=True)
sizes = sorted(df['n'].unique())


def wins_table(d):
    t = (d.groupby(['distribution', 'n'])['best_algorithm'].value_counts(normalize=True)
         .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0) * 100).round(1)
    t.insert(0, 'N', d.groupby(['distribution', 'n']).size())
    return t


wins, wins_strict = wins_table(df), wins_table(df[df['n_tied'] == 1])
for c in CLASSES:  # brecha relativa frente al mejor de los tres
    df[f'gap_{c}'] = (df[f'cost_{c}'] - df[[f'cost_{a}' for a in CLASSES]].min(axis=1)) / df[[f'cost_{a}' for a in CLASSES]].min(axis=1)
gaps = (df.groupby(['distribution', 'n'])[[f'gap_{c}' for c in CLASSES]].mean() * 100).round(3)

caches = pd.concat([pd.read_csv(f'{OUT_DIR}/cache_{c}{tag}.csv').assign(algorithm=c) for c in CLASSES])
times = (caches.merge(df[['instance_id', 'n']], on='instance_id')
         .pivot_table(index='n', columns='algorithm', values='seconds', aggfunc='mean')[CLASSES])

for name, t in [('wins_by_dist_n', wins), ('wins_strict_by_dist_n', wins_strict),
                ('gap_pct_by_dist_n', gaps), ('mean_seconds_by_n', times)]:
    t.to_csv(f'{OUT_DIR}/{name}{tag}.csv')
    print(f'\n== {name} ==\n{t}')

# 1) barras apiladas de % de victorias por n, una por distribución
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, d in zip(axes, DISTS):
    w = wins.loc[d][CLASSES]
    w.plot(kind='bar', stacked=True, ax=ax, legend=False)
    ax.set_title(d); ax.set_xlabel('n')
axes[0].set_ylabel('% de victorias')
fig.legend(CLASSES, loc='lower center', ncol=3, fontsize=9)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f'{FIG}/wins_by_n_distribution{tag}.png', dpi=200); plt.close()

# 2) mapa de calor: brecha media (%) de cada algoritmo, distribución x n
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, c in zip(axes, CLASSES):
    g = gaps[f'gap_{c}'].unstack().reindex(index=DISTS, columns=sizes)
    im = ax.imshow(g.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(sizes)), sizes); ax.set_yticks(range(3), DISTS)
    for i in range(3):
        for j in range(len(sizes)):
            ax.text(j, i, f'{g.values[i, j]:.1f}', ha='center', va='center', color='w', fontsize=7)
    ax.set_title(f'Brecha media (%) – {c}', fontsize=9); ax.set_xlabel('n')
    fig.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout(); plt.savefig(f'{FIG}/gap_heatmap{tag}.png', dpi=200); plt.close()

# 3) tiempo medio por algoritmo y n
ax = times.plot(marker='o', logy=True, figsize=(6, 4))
ax.set_xlabel('n'); ax.set_ylabel('segundos por instancia (media)'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{FIG}/mean_time_by_n{tag}.png', dpi=200); plt.close()
print(f'\nFiguras en {FIG}/ (con ~18 instancias por celda, los % por celda son ruidosos)')
