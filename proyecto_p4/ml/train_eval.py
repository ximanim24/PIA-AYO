# ml/train_eval.py
# Flujo completo de la Persona 4: baseline, CV sobre train (Árbol y KNN),
# selección por CV, evaluación en test (una sola vez), regret, importancias,
# correlación entre features, ablación por grupos y verificación de colapso.
import argparse
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from ml.features import CLASSES, NUM, CAT, FEATURES, load, check_labeled

SEED = 0


def preproc(scale, num=NUM, cat=CAT):
    parts = []
    if num:
        parts.append(('num', StandardScaler() if scale else 'passthrough', num))
    if cat:
        parts.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat))
    return ColumnTransformer(parts)


def regret(df, preds):
    # regret = costo del algoritmo elegido - costo del mejor real
    costs = df[[f'cost_{c}' for c in CLASSES]].to_numpy()
    idx = np.array([CLASSES.index(p) for p in preds])
    chosen = costs[np.arange(len(df)), idx]
    best = costs.min(axis=1)
    return chosen - best, (chosen - best) / best


def evaluate(name, model, te, outdir, dominant):
    X, y = te[FEATURES], te['best_algorithm']
    pred = model.predict(X)
    reg, reg_rel = regret(te, pred)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        cmap='Blues', xticks_rotation=20)
    plt.title(f'Matriz de confusión en test – {name}')
    plt.tight_layout()
    plt.savefig(f'{outdir}/confusion_matrix_{name}.png', dpi=200)
    plt.close()
    # casos donde la clase dominante NO era la mejor: ¿el modelo los detecta?
    non_dom = (y != dominant).to_numpy()
    # Incertidumbre: IC 95% bootstrap de la accuracy en test (semilla fija)
    ok = (pred == y.to_numpy()).astype(float)
    brng = np.random.default_rng(SEED)
    boots = [ok[brng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    # Solo análisis (NUNCA feature): accuracy según empates en la etiqueta
    by_ties = {}
    if 'n_tied' in te.columns:
        for k, g in te.groupby('n_tied'):
            m = (te['n_tied'] == k).to_numpy()
            by_ties[int(k)] = {'n': int(m.sum()),
                               'accuracy': float(ok[m].mean()),
                               'regret_mean': float(reg[m].mean())}
    return {
        'accuracy_ci95': ci,
        'accuracy_by_n_tied': by_ties,
        'accuracy': accuracy_score(y, pred),
        'f1_macro': f1_score(y, pred, average='macro', labels=CLASSES),
        'confusion_matrix': cm.tolist(),
        'report': classification_report(y, pred, labels=CLASSES,
                                        output_dict=True, zero_division=0),
        'pred_counts': pd.Series(pred).value_counts().to_dict(),
        'regret_mean': float(reg.mean()),
        'regret_median': float(np.median(reg)),
        'regret_max': float(reg.max()),
        'regret_rel_mean': float(reg_rel.mean()),
        'regret_by_n': {int(k): float(v) for k, v in
                        pd.Series(reg).groupby(te['n'].to_numpy()).mean().items()},
        'dominant_class': dominant,
        'n_non_dominant': int(non_dom.sum()),
        'recall_non_dominant': float((pred[non_dom] == y.to_numpy()[non_dom]).mean())
        if non_dom.any() else None,
        'pct_pred_non_dominant': float((pred != dominant).mean()),
    }


def correlation(tr, outdir):
    corr = tr[NUM].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.to_numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM)))
    ax.set_yticks(range(len(NUM)))
    ax.set_xticklabels(NUM, rotation=35, ha='right')
    ax.set_yticklabels(NUM)
    for i in range(len(NUM)):
        for j in range(len(NUM)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=8)
    fig.colorbar(im)
    ax.set_title('Correlación entre features numéricas (train)')
    fig.tight_layout()
    fig.savefig(f'{outdir}/feature_correlation.png', dpi=200)
    plt.close(fig)
    return corr.round(4).to_dict()


def build_clf(kind, params):
    p = {k.replace('clf__', ''): v for k, v in params.items()}
    if kind == 'tree':
        return DecisionTreeClassifier(random_state=SEED, **p), False
    return KNeighborsClassifier(**p), True


def ablation(kind, params, tr, cv, outdir):
    """Reentrena con CV sobre train quitando un grupo a la vez (hiperparámetros
    fijos del modelo elegido) y compara la accuracy de CV."""
    ytr = tr['best_algorithm']
    NUM4 = [c for c in NUM if c != 'n']
    sets = {
        'todas': (NUM, CAT),
        'sin n': ([c for c in NUM if c != 'n'], CAT),
        'sin distribution': (NUM, []),
        'sin las 4 numéricas': (['n'], CAT),
    }
    out = {}
    for name, (num, cat) in sets.items():
        clf, scale = build_clf(kind, params)
        pipe = Pipeline([('prep', preproc(scale, num, cat)), ('clf', clf)])
        s = cross_val_score(pipe, tr[num + cat], ytr, cv=cv, scoring='accuracy')
        out[name] = {'cv_acc_mean': float(s.mean()), 'cv_acc_std': float(s.std())}
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(out)
    ax.bar(ks, [out[k]['cv_acc_mean'] for k in ks],
           yerr=[out[k]['cv_acc_std'] for k in ks], capsize=4, color='#4c78a8')
    ax.set_ylabel('Accuracy CV (train)')
    ax.set_title(f'Ablación por grupos – modelo {kind}')
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(f'{outdir}/ablation.png', dpi=200)
    plt.close(fig)
    return out


def main(suffix, outdir):
    tr, te = load(suffix)
    problemas = check_labeled(tr, te)
    if problemas:
        print('AVISO – revisa las etiquetas:', *problemas, sep='\n  - ')
    print(f'train={len(tr)} filas, test={len(te)} filas')

    Xtr, ytr = tr[FEATURES], tr['best_algorithm']
    n_splits = int(max(2, min(5, ytr.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}

    tree = GridSearchCV(
        Pipeline([('prep', preproc(False)),
                  ('clf', DecisionTreeClassifier(random_state=SEED))]),
        {'clf__max_depth': [2, 3, 4, 5, 6, 8, None],
         'clf__min_samples_leaf': [1, 3, 5, 10],
         'clf__criterion': ['gini', 'entropy'],
         'clf__class_weight': [None, 'balanced']},
        cv=cv, scoring=scoring, refit='acc')
    knn = GridSearchCV(
        Pipeline([('prep', preproc(True)), ('clf', KNeighborsClassifier())]),
        {'clf__n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
         'clf__weights': ['uniform', 'distance'],
         'clf__p': [1, 2]},
        cv=cv, scoring=scoring, refit='acc')
    models = {'tree': tree.fit(Xtr, ytr), 'knn': knn.fit(Xtr, ytr)}

    # Selección SOLO por CV (nunca por test)
    chosen = max(models, key=lambda k: models[k].best_score_)
    base = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr)
    dominant = str(base.classes_[base.class_prior_.argmax()])

    allv = pd.concat([ytr, te['best_algorithm']])
    results = {
        'n_splits_cv': n_splits, 'seed': SEED, 'chosen_by_cv': chosen,
        'baseline_majority_class': dominant,
        'class_balance_train': ytr.value_counts().to_dict(),
        'class_balance_test': te['best_algorithm'].value_counts().to_dict(),
        'class_balance_global_pct': (allv.value_counts(normalize=True) * 100)
        .round(2).to_dict(),
        'baseline': evaluate('baseline', base, te, outdir, dominant)}
    for name, m in models.items():
        results[name] = {
            'best_params': m.best_params_,
            'cv_accuracy': float(m.best_score_),
            'cv_accuracy_std': float(m.cv_results_['std_test_acc'][m.best_index_]),
            'cv_f1_macro': float(m.cv_results_['mean_test_f1'][m.best_index_]),
            'test': evaluate(name, m.best_estimator_, te, outdir, dominant)}

    best = models[chosen].best_estimator_
    # Copia con el nombre que pide la guía para el modelo elegido
    import shutil
    shutil.copy(f'{outdir}/confusion_matrix_{chosen}.png',
                f'{outdir}/confusion_matrix.png')

    # Importancia por permutación (variable original, sobre test)
    pi = permutation_importance(best, te[FEATURES], te['best_algorithm'],
                                n_repeats=30, random_state=SEED,
                                scoring='accuracy')
    results['permutation_importance'] = {
        f: {'mean': float(m), 'std': float(s)}
        for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)}
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURES[i] for i in order], pi.importances_mean[order],
            xerr=pi.importances_std[order], color='#59a14f')
    ax.set_xlabel('Caída de accuracy al permutar (test, 30 repeticiones)')
    ax.set_title(f'Importancia por permutación – {chosen}')
    fig.tight_layout()
    fig.savefig(f'{outdir}/permutation_importance.png', dpi=200)
    plt.close(fig)

    # Importancia y reglas del árbol
    t = models['tree'].best_estimator_
    names = list(t.named_steps['prep'].get_feature_names_out())
    results['tree_rules'] = export_text(t.named_steps['clf'], feature_names=names)
    results['tree_feature_importances'] = dict(
        zip(names, map(float, t.named_steps['clf'].feature_importances_)))

    # Colinealidad y ablación (solo con train + CV)
    results['feature_correlation'] = correlation(tr, outdir)
    results['ablation_cv'] = ablation(chosen, models[chosen].best_params_,
                                      tr, cv, outdir)

    joblib.dump(best, f'{outdir}/model_{chosen}.joblib')
    with open(f'{outdir}/ml_results{suffix}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    b, c = results['baseline'], results[chosen]['test']
    print(f"\nModelo elegido por CV: {chosen} "
          f"(cv_acc tree={results['tree']['cv_accuracy']:.3f}, "
          f"knn={results['knn']['cv_accuracy']:.3f})")
    print(f"Baseline ({dominant}): acc={b['accuracy']:.3f} "
          f"regret={b['regret_mean']:.4f}")
    print(f"Modelo {chosen}: acc={c['accuracy']:.3f} "
          f"f1_macro={c['f1_macro']:.3f} regret={c['regret_mean']:.4f}")
    print('Predicciones por clase:', c['pred_counts'])
    print(f"Recall donde '{dominant}' NO gana:", c['recall_non_dominant'])
    print('Ablación (acc CV):', {k: round(v['cv_acc_mean'], 3)
                                 for k, v in results['ablation_cv'].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='', help="'' = reales, '_mock' = falsas")
    ap.add_argument('--outdir', default='report')
    a = ap.parse_args()
    main(a.suffix, a.outdir)
