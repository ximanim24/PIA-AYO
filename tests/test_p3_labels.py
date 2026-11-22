import pandas as pd
from label_generation.run_labels import pick_best, label


def test_pick_best_strict_winner():
    assert pick_best({'2opt_hill_climbing': 3.0, 'sa_2opt': 2.0, 'mst_2opt': 4.0}) == ('sa_2opt', 1)


def test_pick_best_tie_uses_priority_and_tolerance():
    # empate (dentro de 1e-9) entre sa y mst -> gana mst por orden de prioridad
    assert pick_best({'2opt_hill_climbing': 5.0, 'sa_2opt': 2.0, 'mst_2opt': 2.0 + 1e-12}) == ('mst_2opt', 2)
    # triple empate -> gana 2-opt
    assert pick_best({'2opt_hill_climbing': 1.0, 'sa_2opt': 1.0, 'mst_2opt': 1.0}) == ('2opt_hill_climbing', 3)


def test_label_adds_columns_and_keeps_originals():
    split = pd.DataFrame({'instance_id': ['a', 'b'], 'n': [5, 10]})
    mk = lambda v: pd.DataFrame({'instance_id': ['a', 'b'], 'cost': v, 'seconds': 0.1})
    caches = {'2opt_hill_climbing': mk([1.0, 3.0]), 'sa_2opt': mk([1.0, 2.0]), 'mst_2opt': mk([2.0, 2.5])}
    out = label(split, caches)
    assert list(out['best_algorithm']) == ['2opt_hill_climbing', 'sa_2opt']
    assert list(out['n_tied']) == [2, 1] and list(out['n']) == [5, 10]


def test_label_column_order_matches_assignment():
    split = pd.DataFrame({'instance_id': ['a'], 'n': [5]})
    mk = lambda v: pd.DataFrame({'instance_id': ['a'], 'cost': v, 'seconds': 0.1})
    out = label(split, {'2opt_hill_climbing': mk([1.0]), 'mst_2opt': mk([2.0]), 'sa_2opt': mk([3.0])})
    assert list(out.columns) == ['instance_id', 'n', 'cost_2opt_hill_climbing', 'cost_sa_2opt', 'cost_mst_2opt',
                                 'best_algorithm', 'n_tied']


def test_run_one_rejects_inconsistent_cost_or_invalid_tour():
    import pytest
    from label_generation.run_labels import run_one

    class BadCost:
        @staticmethod
        def solve(coords, seed=0):
            return list(range(len(coords))), 0.0  # costo falso

    class BadTour:
        @staticmethod
        def solve(coords, seed=0):
            return [0] * len(coords), 1.0  # no es permutación

    with pytest.raises(ValueError):
        run_one(BadCost, 'inst_0007')
    with pytest.raises(ValueError):
        run_one(BadTour, 'inst_0007')
