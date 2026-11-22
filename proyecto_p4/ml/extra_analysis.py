# ml/extra_analysis.py
# Análisis COMPLEMENTARIO sobre el modelo ya elegido por CV (no decide nada nuevo):
#   - IC 95% bootstrap pareado de la diferencia de regret (modelo - baseline)
#   - dónde se concentra el regret (distribución y tamaño)
#   - victorias estrictas (sin empates) vs. con empates, usando n_tied SOLO para describir
# Uso: python -m ml.extra_analysis   (después de python -m ml.train_eval)
import glob
import json

import joblib
import numpy as np
import pandas as pd

from ml.features import CLASSES, FEATURES, load
from ml.train_eval import regret, SEED

OUT = 'report'


def main():
    tr, te = load('')
    with open(f'{OUT}/ml_results.json') as f:
        res = json.load(f)
    chosen = res['chosen_by_cv']
    model = joblib.load(glob.glob(f'{OUT}/model_{chosen}.joblib')[0])
    dominant = res['baseline_majority_class']

    pred = model.predict(te[FEATURES])
    base = np.array([dominant] * len(te))
    r_m, _ = regret(te, pred)
    r_b, _ = regret(te, base)

    # bootstrap pareado: mismas instancias remuestreadas para modelo y baseline
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(5000):
        i = rng.integers(0, len(te), len(te))
        diffs.append(r_m[i].mean() - r_b[i].mean())
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    def group(col):
        g = pd.DataFrame({'k': col, 'm': r_m, 'b': r_b}).groupby('k')
        return {str(k): {'n': int(len(v)), 'regret_modelo': float(v['m'].mean()),
                         'regret_baseline': float(v['b'].mean())} for k, v in g}

    big = (te['n'] >= 60).to_numpy()
    ok_m = (pred == te['best_algorithm'].to_numpy())
    ok_b = (base == te['best_algorithm'].to_numpy())
    strict = (te['n_tied'] == 1).to_numpy()

    # top-5 instancias con más regret del baseline (para ver qué los explica)
    top = te.assign(regret_base=r_b, regret_modelo=r_m).nlargest(
        5, 'regret_base')[['instance_id', 'n', 'distribution', 'best_algorithm',
                           'regret_base', 'regret_modelo']]

    allv = pd.concat([tr, te], ignore_index=True)
    strict_all = allv[allv['n_tied'] == 1]['best_algorithm'].value_counts()

    out = {
        'modelo': chosen,
        'regret_diff_mean_modelo_menos_baseline': float(r_m.mean() - r_b.mean()),
        'regret_diff_ci95_bootstrap_pareado': ci,
        'regret_por_distribucion': group(te['distribution']),
        'regret_por_n': group(te['n']),
        'regret_n_ge_60': {'n_instancias': int(big.sum()),
                           'modelo': float(r_m[big].mean()),
                           'baseline': float(r_b[big].mean())},
        'regret_n_lt_60': {'n_instancias': int((~big).sum()),
                           'modelo': float(r_m[~big].mean()),
                           'baseline': float(r_b[~big].mean())},
        'top5_regret_baseline': top.to_dict(orient='records'),
        'accuracy_solo_victorias_estrictas': {
            'n_instancias': int(strict.sum()),
            'modelo': float(ok_m[strict].mean()),
            'baseline': float(ok_b[strict].mean())},
        'accuracy_con_empates_n_tied_ge_2': {
            'n_instancias': int((~strict).sum()),
            'modelo': float(ok_m[~strict].mean()),
            'baseline': float(ok_b[~strict].mean())},
        'victorias_estrictas_500_instancias': strict_all.to_dict(),
        'empates_triples_500_instancias': int((allv['n_tied'] == 3).sum()),
    }
    with open(f'{OUT}/extra_analysis.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in (
        'regret_diff_mean_modelo_menos_baseline',
        'regret_diff_ci95_bootstrap_pareado', 'regret_n_ge_60', 'regret_n_lt_60',
        'accuracy_solo_victorias_estrictas', 'accuracy_con_empates_n_tied_ge_2',
        'victorias_estrictas_500_instancias')}, indent=1, ensure_ascii=False))
    print(top.to_string(index=False))
    print('por distribución:', json.dumps(out['regret_por_distribucion']))


if __name__ == '__main__':
    main()
