# ml/features.py
# Definición de features, carga de datos y (opcional) features propias.
import numpy as np
import pandas as pd

CLASSES = ['2opt_hill_climbing', 'sa_2opt', 'mst_2opt']
NUM = ['n', 'mean_pairwise_dist', 'std_coords', 'mean_centroid_dist',
       'mean_nn_dist']
CAT = ['distribution']
FEATURES = NUM + CAT

# NUNCA como feature (fuga de información): instance_id, cost_*, best_algorithm, n_tied
LEAKY = ['instance_id', 'cost_2opt_hill_climbing', 'cost_sa_2opt',
         'cost_mst_2opt', 'best_algorithm', 'n_tied']


def load(suffix=''):
    """suffix='' -> etiquetas reales; '_mock' -> etiquetas falsas."""
    tr = pd.read_csv(f'data/labeled/train_labeled{suffix}.csv')
    te = pd.read_csv(f'data/labeled/test_labeled{suffix}.csv')
    return tr, te


def check_labeled(tr, te):
    """Chequeos del Paso 8 de la guía (sanidad de las etiquetas de P3)."""
    problemas = []
    for nombre, df in (('train', tr), ('test', te)):
        cost_cols = [c for c in df.columns if c.startswith('cost_')]
        if df[cost_cols].isna().any().any():
            problemas.append(f'{nombre}: hay NaN en cost_*')
        malos = set(df['best_algorithm'].unique()) - set(CLASSES)
        if malos:
            problemas.append(f'{nombre}: best_algorithm con valores inválidos {malos}')
        faltan = [c for c in FEATURES if c not in df.columns]
        if faltan:
            problemas.append(f'{nombre}: faltan columnas {faltan}')
    return problemas


# --- Opcional (Paso 7): requiere tsp_utils.py de P1 ---------------------------
def extra_features(instance_id):
    from tsp_utils import load_instance, dist_matrix  # noqa: solo si P1 lo entrega
    c = load_instance(instance_id)
    D = dist_matrix(c)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    border = np.minimum.reduce([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]])
    return {
        'std_nn_dist': float(nn.std()),
        'mean_border_dist': float(border.mean()),
        'frac_near_border': float((border < 0.1).mean()),
        'bbox_area': float(np.ptp(c[:, 0]) * np.ptp(c[:, 1])),
    }
