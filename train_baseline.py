"""
Primer modelo (baseline): regresión logística con las diferencias A - B, añadiendo
grupos de características uno a uno para ver cuánto aporta cada grupo.

Entrada : model_dataset.csv
Ejecutar: python train_baseline.py

El modelo se entrena SOLO con 'train'. Se compara en 'val' mientras se prueban
cosas. 'test' se mira al final, una sola vez, para no engañarnos a nosotros mismos.
"""
import numpy as np
import pandas as pd

from common import FEATURES, SymmetricModel, accuracy, brier, logloss


def main():
    d = pd.read_csv("model_dataset.csv", parse_dates=["match_date"])
    train, val, test = (d[d["split"] == s] for s in ["train", "val", "test"])
    rows = []

    def add(name, fn):
        r = {"modelo": name}
        for sname, part in [("val", val), ("test", test)]:
            y, p = part["target"].to_numpy(), fn(part)
            r.update({f"{sname}_acierto": accuracy(y, p), f"{sname}_logloss": logloss(y, p),
                      f"{sname}_brier": brier(y, p)})
        rows.append(r)

    # --- Referencias sin entrenar ---
    add("Siempre 50% (moneda)", lambda p: np.full(len(p), 0.5))
    add("Elo (fórmula, sin entrenar)", lambda p: p["elo_prob_A"].to_numpy())
    # empate de ranking (p. ej. los dos sin ranking) -> 50 %
    add("Mejor ranking (95% / 5%)",
        lambda p: np.select([p["log_rank_diff"] < 0, p["log_rank_diff"] > 0], [0.95, 0.05], 0.5))

    # --- Regresión logística con distintos grupos de características ---
    base = ["elo_diff", "elo_surf_diff", "log_rank_diff", "age_diff", "ht_diff", "lefty_diff", "exp_diff"]
    serve = ["spw_diff", "rpw_diff", "ace_diff", "df_diff", "fin_diff"]
    history = ["form_diff", "rest_diff", "layoff_diff", "games14_diff", "h2h_diff"]
    sets = {
        "LogReg: solo Elo": ["elo_diff"],
        "LogReg: Elo + Elo superficie": ["elo_diff", "elo_surf_diff"],
        "LogReg: base (Elo, ranking, perfil)": base,
        "  + al mejor de 5 (elo_diff_bo5)": base + ["elo_diff_bo5"],
        "  + forma reciente": base + ["form_diff"],
        "  + descanso, inactividad, cansancio": base + ["rest_diff", "layoff_diff", "games14_diff"],
        "  + cara a cara": base + ["h2h_diff"],
        "  + saque y resto": base + serve,
        "LogReg: TODO sin saque/resto": base + ["elo_diff_bo5"] + history,
        "LogReg: TODO": FEATURES,
    }
    models = {}
    for name, feats in sets.items():
        # sin intercepto: si A y B se intercambian, la probabilidad se invierte (simetría)
        mdl = SymmetricModel(name, "logreg", features=feats).fit(train)
        models[name] = mdl
        add(name, mdl.predict)

    res = pd.DataFrame(rows).set_index("modelo")
    pd.set_option("display.width", 200)
    print(f"Entrenamiento: {len(train):,} partidos | Validación: {len(val):,} | Test: {len(test):,}\n")
    print("VALIDACIÓN (2022-2023)")
    print(res[["val_acierto", "val_logloss", "val_brier"]].round(4).to_string())
    print("\nTEST (2024 en adelante)")
    print(res[["test_acierto", "test_logloss", "test_brier"]].round(4).to_string())

    # --- Coeficiente del Elo: ¿concuerda con la fórmula teórica? ---
    pipe = models["LogReg: solo Elo"].model
    raw_coef = pipe[-1].coef_[0][0] / pipe[0].scale_[0]
    print(f"\nCoeficiente aprendido de elo_diff: {raw_coef:.5f}  |  el de la fórmula teórica: {np.log(10) / 400:.5f}")

    pipe = models["LogReg: TODO"].model
    imp = pd.Series(pipe[-1].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print("\nPeso de cada característica (valores estandarizados; mayor valor absoluto = más influencia):")
    print(imp.round(3).to_string())


if __name__ == "__main__":
    main()
