"""Parte 2 del PIA: clasificador que predice best_algorithm (Árbol de Decisión y KNN).

    python -m ml_model.train_eval [--tag ""]

- Características permitidas: n, distribution y las 4 numéricas provistas. Nunca costos ni n_tied (fuga).
- Hiperparámetros: validación cruzada estratificada de 5 pliegues SOLO sobre train.
- El modelo elegido (mayor accuracy de CV) y el otro se evalúan en test una sola vez.
- Métricas: accuracy vs baseline (clase mayoritaria de train), matriz de confusión, regret, importancia de características.
Salidas en results/.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from label_generation.run_labels import OUT_DIR, PRIORITY as CLASSES, TOL

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='')
tag = ap.parse_args().tag
RES = 'results'
os.makedirs(RES, exist_ok=True)
SEED = 42
NUM = ['n', 'mean_pairwise_dist', 'std_coords', 'mean_centroid_dist', 'mean_nn_dist']
CAT = ['distribution']
FEATURES = NUM + CAT
FORBIDDEN = [c for c in ['n_tied', 'best_algorithm'] if c in FEATURES] + [c for c in FEATURES if c.startswith('cost_')]
assert not FORBIDDEN, f'fuga de información: {FORBIDDEN}'

tr = pd.read_csv(f'{OUT_DIR}/train_labeled{tag}.csv')
te = pd.read_csv(f'{OUT_DIR}/test_labeled{tag}.csv')
Xtr, ytr, Xte, yte = tr[FEATURES], tr['best_algorithm'], te[FEATURES], te['best_algorithm']
cost_te = te[[f'cost_{c}' for c in CLASSES]].to_numpy()
best_te = cost_te.min(axis=1)


def prep(scale):
    return ColumnTransformer([('num', StandardScaler() if scale else 'passthrough', NUM),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), CAT)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
searches = {
    'DecisionTree': GridSearchCV(
        Pipeline([('prep', prep(False)), ('clf', DecisionTreeClassifier(random_state=SEED))]),
        {'clf__max_depth': [2, 3, 4, 5, 6, 8, None], 'clf__min_samples_leaf': [1, 2, 5, 10, 20],
         'clf__class_weight': [None, 'balanced']}, cv=cv, scoring='accuracy'),
    'KNN': GridSearchCV(
        Pipeline([('prep', prep(True)), ('clf', KNeighborsClassifier())]),
        {'clf__n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 31], 'clf__weights': ['uniform', 'distance']},
        cv=cv, scoring='accuracy'),
}
for name, g in searches.items():
    g.fit(Xtr, ytr)
chosen = max(searches, key=lambda k: searches[k].best_score_)  # se elige ANTES de mirar el test

base_class = ytr.value_counts().idxmax()
cv_baseline = float(cross_val_score(DummyClassifier(strategy='most_frequent'), Xtr, ytr, cv=cv, scoring='accuracy').mean())
baseline_acc = float((yte == base_class).mean())
idx = {c: i for i, c in enumerate(CLASSES)}


def regret_of(pred):
    chosen_cost = cost_te[np.arange(len(pred)), [idx[p] for p in pred]]
    return chosen_cost - best_te


summary, out = [], {}
reg_df = te[['instance_id', 'n', 'distribution', 'best_algorithm', 'n_tied']].copy()
for name, g in searches.items():
    pred = g.predict(Xte)
    reg = regret_of(pred)
    rel = reg / best_te * 100
    hc_worse = (cost_te[:, 1] < cost_te[:, 0] - TOL) | (cost_te[:, 2] < cost_te[:, 0] - TOL)  # SA o MST superan a HC
    summary.append({'model': name, 'chosen': name == chosen, 'best_params': json.dumps({k.split('__')[1]: v for k, v in g.best_params_.items()}),
                    'cv_accuracy': g.best_score_, 'cv_baseline_accuracy': cv_baseline, 'test_accuracy': accuracy_score(yte, pred),
                    'test_accuracy_ci95_pm': 1.96 * float(np.sqrt(accuracy_score(yte, pred) * (1 - accuracy_score(yte, pred)) / len(yte))),
                    'baseline_accuracy': baseline_acc, 'regret_mean': reg.mean(), 'regret_median': float(np.median(reg)),
                    'regret_max': reg.max(), 'regret_rel_mean_pct': rel.mean(), 'pct_instancias_regret_cero': float((reg <= TOL).mean() * 100),
                    'n_instancias_SA_o_MST_superan_HC': int(hc_worse.sum()),
                    'de_ellas_modelo_no_elige_HC': int((pred[hc_worse] != '2opt_hill_climbing').sum()),
                    'de_ellas_regret_cero': int((reg[hc_worse] <= TOL).sum())})
    reg_df[f'pred_{name}'], reg_df[f'regret_{name}'] = pred, reg
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    pd.DataFrame(cm, index=[f'real_{c}' for c in CLASSES], columns=[f'pred_{c}' for c in CLASSES]).to_csv(f'{RES}/confusion_matrix_{name}{tag}.csv')
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3), CLASSES, rotation=20, fontsize=7); ax.set_yticks(range(3), CLASSES, fontsize=7)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real'); ax.set_title(f'{name}: matriz de confusión (test)', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{RES}/confusion_matrix_{name}{tag}.png', dpi=200); plt.close()
    out[name] = {'recall_por_clase': dict(zip(CLASSES, recall_score(yte, pred, labels=CLASSES, average=None, zero_division=0).round(3))),
                 'predicciones_por_clase': pd.Series(pred).value_counts().reindex(CLASSES, fill_value=0).to_dict()}

# baseline "siempre la mayoritaria": su regret
reg_base = regret_of([base_class] * len(yte))
summary.append({'model': f'Baseline (siempre {base_class})', 'chosen': False, 'best_params': '', 'cv_accuracy': np.nan, 'cv_baseline_accuracy': cv_baseline, 'test_accuracy': baseline_acc,
                'test_accuracy_ci95_pm': 1.96 * float(np.sqrt(baseline_acc * (1 - baseline_acc) / len(yte))),
                'baseline_accuracy': baseline_acc, 'regret_mean': reg_base.mean(), 'regret_median': float(np.median(reg_base)),
                'regret_max': reg_base.max(), 'regret_rel_mean_pct': (reg_base / best_te * 100).mean(),
                'pct_instancias_regret_cero': float((reg_base <= TOL).mean() * 100)})
pd.DataFrame(summary).to_csv(f'{RES}/model_summary{tag}.csv', index=False)
reg_df.to_csv(f'{RES}/regret_by_instance{tag}.csv', index=False)
json.dump(out, open(f'{RES}/model_details{tag}.json', 'w'), indent=1, ensure_ascii=False)
pd.concat([pd.DataFrame(g.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].assign(model=n)
           for n, g in searches.items()]).to_csv(f'{RES}/cv_grid{tag}.csv', index=False)

# accuracy por distribución y por banda de tamaño (modelo elegido)
best_pred = searches[chosen].predict(Xte)
grp = te.assign(ok=(best_pred == yte).values, banda=pd.cut(te['n'], [0, 15, 40, 100], labels=['n<=15', '20<=n<=40', 'n>=60']))
pd.concat([grp.groupby('distribution')['ok'].agg(['mean', 'count']).rename_axis('grupo'),
           grp.groupby('banda', observed=True)['ok'].agg(['mean', 'count']).rename_axis('grupo')]).to_csv(f'{RES}/accuracy_by_group{tag}.csv')

# importancia de características (permutación sobre columnas originales, en test) + importancia propia del árbol
imp = permutation_importance(searches[chosen].best_estimator_, Xte, yte, n_repeats=50, random_state=SEED, scoring='accuracy')
imp_df = pd.DataFrame({'feature': FEATURES, 'perm_importance_mean': imp.importances_mean, 'perm_importance_std': imp.importances_std})
tree_imp = None
if 'DecisionTree' in searches:
    dt = searches['DecisionTree'].best_estimator_
    names = list(dt.named_steps['prep'].get_feature_names_out())
    ti = pd.Series(dt.named_steps['clf'].feature_importances_, index=[n.split('__')[1] for n in names])
    ti_group = ti.groupby(lambda s: 'distribution' if s.startswith('distribution_') else s).sum()
    imp_df['tree_importance'] = imp_df['feature'].map(ti_group).fillna(0)
    open(f'{RES}/tree_rules{tag}.txt', 'w', encoding='utf-8').write(
        export_text(dt.named_steps['clf'], feature_names=[n.split('__')[1] for n in names]))
imp_df.sort_values('perm_importance_mean', ascending=False).to_csv(f'{RES}/feature_importance{tag}.csv', index=False)
fig, ax = plt.subplots(figsize=(6, 3.4))
d = imp_df.sort_values('perm_importance_mean')
ax.barh(d['feature'], d['perm_importance_mean'], xerr=d['perm_importance_std'])
ax.set_xlabel('Caída de accuracy al permutar la característica (test)'); ax.set_title(f'Importancia por permutación – {chosen}', fontsize=9)
plt.tight_layout(); plt.savefig(f'{RES}/feature_importance{tag}.png', dpi=200); plt.close()

print(pd.DataFrame(summary).round(4).T.to_string())
print('\nElegido por CV:', chosen)
print(json.dumps(out, indent=1, ensure_ascii=False))
print(imp_df.round(4).to_string())
