"""
Ejecuta todo el pipeline en orden.

  python run_pipeline.py                  # todo, incluida la descarga
  python run_pipeline.py --skip-download  # usa los CSV ya descargados
  python run_pipeline.py --from compute_elo
  python run_pipeline.py --baseline       # incluye también train_baseline.py (comparativa de features)
  python run_pipeline.py --backtest       # al final, simula los torneos reales de test (≈ 2-3 min)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["data_set", "clean_matches", "compute_elo", "add_features", "add_serve_stats",
         "build_model_dataset", "train_best"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="no volver a descargar los datos")
    ap.add_argument("--from", dest="start", choices=STEPS, help="empezar en este paso")
    ap.add_argument("--baseline", action="store_true", help="ejecutar también train_baseline.py")
    ap.add_argument("--backtest", action="store_true", help="ejecutar también backtest_tournaments.py al final")
    args = ap.parse_args()

    steps = STEPS[STEPS.index(args.start):] if args.start else list(STEPS)
    if args.skip_download and "data_set" in steps:
        steps.remove("data_set")
    if args.baseline:
        steps.insert(steps.index("train_best") if "train_best" in steps else len(steps), "train_baseline")
    if args.backtest:
        steps.append("backtest_tournaments")

    t_all = time.time()
    for step in steps:
        print(f"\n{'=' * 70}\n>>> {step}.py\n{'=' * 70}", flush=True)
        t = time.time()
        res = subprocess.run([sys.executable, str(ROOT / f"{step}.py")], cwd=ROOT)
        if res.returncode != 0:
            sys.exit(f"\nFalló {step}.py (código {res.returncode}). Se detiene el pipeline.")
        print(f"--- {step}.py terminado en {time.time() - t:.0f} s", flush=True)
    print(f"\nPipeline completo en {time.time() - t_all:.0f} s.")
    print('Prueba:  python predict.py "Jannik Sinner" "Carlos Alcaraz" --surface Clay --best-of 5')


if __name__ == "__main__":
    main()
