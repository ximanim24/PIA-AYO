"""Corre todo el flujo de la Persona 3, en orden y con un solo comando:
etiquetado -> verificación -> balance de clases -> patrones.

    python -m label_generation.run_all                      # con el paquete algorithms/ (P1 y P2)
    python -m label_generation.run_all --tarea2 RUTA        # con los archivos de la Tarea 2
    python -m label_generation.run_all --stub               # prueba rápida con algoritmos falsos
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--stub', action='store_true')
ap.add_argument('--tarea2', metavar='DIR')
ap.add_argument('--n_jobs', default='-1')
args = ap.parse_args()

tag = ('_stub' if args.stub else '') + ('_t2' if args.tarea2 else '')
label_args = ['--n_jobs', args.n_jobs] + (['--stub'] if args.stub else []) + (['--tarea2', args.tarea2] if args.tarea2 else [])
steps = [('run_labels', label_args), ('check_labels', ['--tag', tag]), ('class_balance', ['--tag', tag]),
         ('pattern_analysis', ['--tag', tag])]
for name, a in steps:
    print(f'\n=== {name} ===', flush=True)
    if subprocess.run([sys.executable, '-m', f'label_generation.{name}'] + a).returncode != 0:
        sys.exit(f'Falló el paso {name}; se detiene.')
print('\nListo. Revisa data/labeled/ y report/')
