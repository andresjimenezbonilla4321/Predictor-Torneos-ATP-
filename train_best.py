"""
Entrena varios modelos, elige el de MENOR LOG-LOSS en validación, lo reentrena con
train + val y lo examina UNA sola vez en test. Después lo reentrena con TODOS los
datos y lo guarda para predecir partidos y torneos futuros.

Por qué log-loss y no acierto: el acierto solo mira si p > 0.5, es más ruidoso y no
premia que las probabilidades sean correctas. Para simular torneos lo que importa es
que un 70 % signifique de verdad un 70 %.

Entrada : model_dataset.csv
Salida  : predictions_best.csv     (una fila por partido de test)
          train_best_report.txt    (el informe que se imprime)
          models/best_model.joblib (modelo final entrenado con todos los datos)

Ejecutar:  python train_best.py        (tarda unos minutos)
"""
import joblib
import numpy as np
import pandas as pd

from common import (FEATURES, MODELS_DIR, CalibratedModel, EnsembleModel, SymmetricModel, accuracy,
                    brier, calibration_table, logloss)

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "model_dataset.csv"
# candidatos de árboles: (profundidad, velocidad de aprendizaje, nº de árboles, mínimo por hoja)
HGB_GRID = [(3, 0.05, 300, 200), (4, 0.03, 400, 200), (6, 0.03, 300, 100), (8, 0.02, 400, 100)]
# -----------------------------------------------------------------


def fit_rows(df):
    """Para ENTRENAR se quitan las retiradas (resultado poco informativo); para evaluar se mantienen."""
    return df[df["is_retirement"] == 0]


def build_candidates():
    logreg = SymmetricModel("LogReg", "logreg")
    trees = [SymmetricModel(f"Árboles prof={g[0]} lr={g[1]} n={g[2]} hoja={g[3]}", "hgb", g) for g in HGB_GRID]
    return logreg, trees


def main():
    lines = []

    def say(text=""):
        print(text)
        lines.append(str(text))

    d = pd.read_csv(INPUT, parse_dates=["match_date"])
    train, val, test = (d[d["split"] == s].copy() for s in ["train", "val", "test"])

    # ---------- 1) Probar candidatos: entrenar con train, medir en val ----------
    say(f"Entrenamiento: {len(train):,} | Validación: {len(val):,} | Test: {len(test):,}")
    say(f"Características ({len(FEATURES)}): {', '.join(FEATURES)}\n")
    say("PASO 1 - Candidatos entrenados con train y medidos en validación (2022-2023)")
    yv = val["target"].to_numpy()
    say(f"  {'Elo (fórmula, sin entrenar)':38s} acierto={accuracy(yv, val['elo_prob_A']):.4f}  "
        f"log-loss={logloss(yv, val['elo_prob_A']):.4f}")

    logreg, trees = build_candidates()
    preds, rows = {}, []
    for c in [logreg] + trees:
        c.fit(train)
        p = c.predict(val)
        preds[c.name] = p
        rows.append((c.name, accuracy(yv, p), logloss(yv, p)))
        say(f"  {c.name:38s} acierto={rows[-1][1]:.4f}  log-loss={rows[-1][2]:.4f}")

    ens_name = "Conjunto (LogReg + media de árboles)"
    p_ens = 0.5 * preds["LogReg"] + 0.5 * np.mean([preds[t.name] for t in trees], axis=0)
    rows.append((ens_name, accuracy(yv, p_ens), logloss(yv, p_ens)))
    say(f"  {ens_name:38s} acierto={rows[-1][1]:.4f}  log-loss={rows[-1][2]:.4f}")

    best_name = min(rows, key=lambda r: r[2])[0]
    se = 100 * np.sqrt(0.65 * 0.35 / len(val))
    say(f"\n  Elegido por menor log-loss en validación: {best_name}")
    say(f"  (El acierto en validación tiene un margen de error de ±{se:.2f} puntos: diferencias")
    say("   menores que eso entre candidatos son ruido, no mejoras reales.)")

    # calibración: cuánto hay que suavizar (a < 1) o afilar (a > 1) las probabilidades del elegido
    p_best = p_ens if best_name == ens_name else preds[best_name]
    calib_a = CalibratedModel.fit_scale(yv, p_best)
    say(f"\n  Calibración ajustada en validación: a = {calib_a:.3f} "
        f"(log-loss {logloss(yv, p_best):.4f} -> {logloss(yv, CalibratedModel.apply(p_best, calib_a)):.4f})")

    def make_best():
        lr, tr = build_candidates()
        if best_name == ens_name:
            return EnsembleModel(ens_name, [lr, EnsembleModel("Árboles", tr)])
        return next(c for c in [lr] + tr if c.name == best_name)

    # el elegido, entrenado SOLO con train y SIN calibrar: lo usa el backtest de validación
    # (backtest_tournaments.py --split val) para ajustar la calibración a nivel de torneo
    if best_name == ens_name:
        train_only = EnsembleModel(ens_name, [logreg, EnsembleModel("Árboles", trees)])
    else:
        train_only = next(c for c in [logreg] + trees if c.name == best_name)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": train_only, "features": FEATURES, "name": train_only.name,
                 "trained_until": str(train["match_date"].max().date())}, MODELS_DIR / "model_train.joblib")

    # ---------- 2) Reentrenar con train + val y examinar en test (una sola vez) ----------
    say("\nPASO 2 - Reentrenar con train + val (2005-2023) y examinar en test (2024 en adelante)")
    model = make_best().fit(fit_rows(pd.concat([train, val])))
    pt_raw = model.predict(test)
    pt = CalibratedModel.apply(pt_raw, calib_a)
    yt = test["target"].to_numpy()
    # este modelo NO ha visto test: es el que usa backtest_tournaments.py para simular torneos de 2024+
    MODELS_DIR.mkdir(exist_ok=True)
    trainval_model = CalibratedModel(model, calib_a)
    joblib.dump({"model": trainval_model, "features": FEATURES, "name": trainval_model.name,
                 "trained_until": str(val["match_date"].max().date())}, MODELS_DIR / "model_trainval.joblib")
    no_ret = (test["is_retirement"] == 0).to_numpy()

    say(f"  {'':34s} {'partidos':>9s} {'acierto':>8s} {'log-loss':>9s} {'Brier':>7s}")
    for label, mk, p in [("Modelo calibrado - todos", np.ones(len(yt), bool), pt),
                         ("Modelo calibrado - sin retiradas", no_ret, pt),
                         ("Modelo sin calibrar - todos", np.ones(len(yt), bool), pt_raw),
                         ("Modelo sin calibrar - sin retiradas", no_ret, pt_raw),
                         ("Elo fórmula - todos", np.ones(len(yt), bool), test["elo_prob_A"].to_numpy())]:
        say(f"  {label:34s} {mk.sum():9d} {accuracy(yt[mk], p[mk]):8.4f} {logloss(yt[mk], p[mk]):9.4f} "
            f"{brier(yt[mk], p[mk]):7.4f}")
    say(f"  Margen de error del acierto en test: ±{100 * np.sqrt(0.65 * 0.35 / len(test)):.2f} puntos")

    # ---------- 3) Calibración ----------
    say("\nPASO 3 - Calibración en test (modelo calibrado): ¿cuando el favorito tiene un X %, gana un X %?")
    say(f"  {'prob. favorito':>15s} {'partidos':>9s} {'predicho':>9s} {'real':>7s} {'dif.':>6s}")
    for a, b, n, pred, real in calibration_table(yt, pt):
        say(f"  {f'{a:.2f}-{b:.2f}':>15s} {n:9d} {pred * 100:8.1f}% {real * 100:6.1f}% {100 * (real - pred):+5.1f}")

    # ---------- 4) Acierto según lo seguro que está el modelo ----------
    conf = np.maximum(pt, 1 - pt)
    hit = (pt > 0.5) == (yt == 1)
    say("\nPASO 4 - Acierto según la confianza del modelo (test)")
    say(f"  {'confianza':>10s} {'partidos':>9s} {'% del total':>12s} {'acierto':>9s}")
    for th in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        mk = conf >= th
        say(f"  {'>= ' + format(th, '.2f'):>10s} {mk.sum():9d} {mk.mean() * 100:11.1f}% {hit[mk].mean() * 100:8.1f}%")

    grp = np.where(test["level_source"] == "main", "Circuito principal", "Challenger")
    say("\nAcierto por nivel y superficie (test):")
    for g in ["Circuito principal", "Challenger"]:
        mk = grp == g
        say(f"  {g:20s} partidos={mk.sum():6d}  acierto={hit[mk].mean() * 100:.1f}%  "
            f"log-loss={logloss(yt[mk], pt[mk]):.4f}")
    gs = (test["tourney_level"] == "G").to_numpy()
    say(f"  {'Grand Slams':20s} partidos={gs.sum():6d}  acierto={hit[gs].mean() * 100:.1f}%")
    for s in ["Hard", "Clay", "Grass"]:
        mk = (test["surface"] == s).to_numpy()
        say(f"  {s:20s} partidos={mk.sum():6d}  acierto={hit[mk].mean() * 100:.1f}%")

    test["prob_A"] = pt.round(4)
    test["confidence"] = conf.round(4)
    test["hit"] = hit.astype(int)
    test["pred_winner"] = np.where(pt > 0.5, test["player_A_name"], test["player_B_name"])
    test["actual_winner"] = np.where(yt == 1, test["player_A_name"], test["player_B_name"])
    test[["match_date", "tourney_name", "level_source", "surface", "round", "player_A_name", "player_B_name",
          "prob_A", "pred_winner", "actual_winner", "hit", "confidence", "is_retirement"]] \
        .to_csv("predictions_best.csv", index=False)

    # ---------- 5) Modelo final con TODOS los datos (para predecir el futuro) ----------
    say("\nPASO 5 - Reentrenar con todos los datos (2005 en adelante) y guardar el modelo final")
    final = CalibratedModel(make_best(), calib_a).fit(fit_rows(d))
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "name": final.name,
                 "trained_until": str(d["match_date"].max().date())}, MODELS_DIR / "best_model.joblib")
    say(f"  Guardado: models/best_model.joblib ({final.name})")

    say("\nGuardado: predictions_best.csv y train_best_report.txt")
    with open("train_best_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
