"""
Primer modelo (baseline): regresión logística con las diferencias A - B.

Entrada : model_dataset.csv
Ejecutar: python train_baseline.py      (necesita: pip install scikit-learn)

El modelo se entrena SOLO con 'train'. Se compara en 'val' mientras se prueban
cosas. 'test' se mira al final, una sola vez, para no engañarnos a nosotros mismos.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = pd.read_csv("model_dataset.csv", parse_dates=["match_date"])
train, val, test = (d[d["split"] == s] for s in ["train", "val", "test"])


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    acc = ((p > 0.5) == (y == 1)).mean()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    brier = ((p - y) ** 2).mean()
    return acc, ll, brier


rows = []


def add(name, fn):
    r = {"modelo": name}
    for sname, part in [("val", val), ("test", test)]:
        acc, ll, br = metrics(part["target"].to_numpy(), fn(part))
        r.update({f"{sname}_acierto": acc, f"{sname}_logloss": ll, f"{sname}_brier": br})
    rows.append(r)


# --- Referencias sin entrenar ---
add("Siempre 50% (moneda)", lambda p: np.full(len(p), 0.5))
add("Elo (fórmula, sin entrenar)", lambda p: p["elo_prob_A"].to_numpy())
add("Mejor ranking (95% / 5%)", lambda p: np.where(p["log_rank_diff"] < 0, 0.95, 0.05))

# --- Regresión logística con distintos grupos de características ---
base = ["elo_diff", "elo_surf_diff", "log_rank_diff", "age_diff", "ht_diff", "lefty_diff", "exp_diff"]
sets = {
    "LogReg: solo Elo": ["elo_diff"],
    "LogReg: Elo + Elo superficie": ["elo_diff", "elo_surf_diff"],
    "LogReg: base (Elo, ranking, perfil)": base,
    "  + forma reciente": base + ["form_diff"],
    "  + descanso y cansancio": base + ["rest_diff", "games14_diff"],
    "  + cara a cara": base + ["h2h_diff"],
    "  + saque y resto": base + ["spw_diff", "rpw_diff", "ace_diff", "df_diff", "fin_diff"],
    "LogReg: TODO sin saque/resto": base + ["form_diff", "rest_diff", "games14_diff", "h2h_diff"],
    "LogReg: TODO": base + ["form_diff", "rest_diff", "games14_diff", "h2h_diff",
                            "spw_diff", "rpw_diff", "ace_diff", "df_diff", "fin_diff"],
}
models = {}
for name, feats in sets.items():
    # sin intercepto: si A y B se intercambian, la probabilidad se invierte (simetría)
    pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(fit_intercept=False, max_iter=1000))
    pipe.fit(train[feats], train["target"])
    models[name] = (pipe, feats)
    add(name, lambda p, pipe=pipe, feats=feats: pipe.predict_proba(p[feats])[:, 1])

res = pd.DataFrame(rows).set_index("modelo")
pd.set_option("display.width", 200)
print(f"Entrenamiento: {len(train):,} partidos | Validación: {len(val):,} | Test: {len(test):,}\n")
print("VALIDACIÓN (2022-2023)")
print(res[["val_acierto", "val_logloss", "val_brier"]].round(4).to_string())
print("\nTEST (2024 en adelante)")
print(res[["test_acierto", "test_logloss", "test_brier"]].round(4).to_string())

# --- Coeficiente del Elo: ¿concuerda con la fórmula teórica? ---
pipe, feats = models["LogReg: solo Elo"]
raw_coef = pipe[-1].coef_[0][0] / pipe[0].scale_[0]
print(f"\nCoeficiente aprendido de elo_diff: {raw_coef:.5f}  |  el de la fórmula teórica: {np.log(10) / 400:.5f}")

pipe, feats = models["LogReg: TODO"]
imp = pd.Series(pipe[-1].coef_[0], index=feats).sort_values(key=abs, ascending=False)
print("\nPeso de cada característica (valores estandarizados; mayor valor absoluto = más influencia):")
print(imp.round(3).to_string())