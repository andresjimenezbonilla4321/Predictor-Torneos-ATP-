"""
Entrena varios modelos, elige el que MÁS ACIERTA en validación, lo reentrena con
train + val y lo examina UNA sola vez en test.

Entrada : model_dataset.csv
Salida  : predictions_best.csv    (una fila por partido de test)
          train_best_report.txt   (el informe que se imprime)

Ejecutar:  python train_best.py        (necesita scikit-learn; tarda 1-3 minutos)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "model_dataset.csv"
FEATURES = ["elo_diff", "elo_surf_diff", "log_rank_diff", "age_diff", "ht_diff", "lefty_diff",
            "exp_diff", "form_diff", "rest_diff", "games14_diff", "h2h_diff",
            "spw_diff", "rpw_diff", "ace_diff", "df_diff", "fin_diff"]
# candidatos de árboles: (profundidad, velocidad de aprendizaje, nº de árboles, mínimo por hoja)
HGB_GRID = [(3, 0.05, 300, 200), (4, 0.03, 400, 200), (6, 0.03, 300, 100), (8, 0.02, 400, 100)]
# -----------------------------------------------------------------

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


d = pd.read_csv(INPUT, parse_dates=["match_date"])
train, val, test = (d[d["split"] == s].copy() for s in ["train", "val", "test"])


def flip(x):
    """Copia del partido con A y B intercambiados (así el modelo no depende de quién es A)."""
    f = x.copy()
    f[FEATURES] = -f[FEATURES]
    f["target"] = 1 - f["target"]
    return f


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def acc(y, p):
    return ((p > 0.5) == (y == 1)).mean()


class Candidate:
    def __init__(self, name, kind, params=None):
        self.name, self.kind, self.params = name, kind, params

    def fit(self, df):
        if self.kind == "logreg":
            self.model = make_pipeline(StandardScaler(with_mean=False),
                                       LogisticRegression(fit_intercept=False, max_iter=2000))
            self.model.fit(df[FEATURES], df["target"])
        else:
            depth, lr, it, leaf = self.params
            both = pd.concat([df, flip(df)])
            self.model = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=it,
                                                        min_samples_leaf=leaf, l2_regularization=5.0, random_state=0)
            self.model.fit(both[FEATURES], both["target"])
        return self

    def predict(self, df):
        X = df[FEATURES]
        # promedio simétrico: P(A gana) = media de [P(A) , 1 - P(B visto como A)]
        return (self.model.predict_proba(X)[:, 1] + 1 - self.model.predict_proba(-X)[:, 1]) / 2


candidates = [Candidate("LogReg", "logreg")]
candidates += [Candidate(f"Árboles prof={g[0]} lr={g[1]} n={g[2]} hoja={g[3]}", "hgb", g) for g in HGB_GRID]

# ---------- 1) Probar candidatos: entrenar con train, medir en val ----------
say(f"Entrenamiento: {len(train):,} | Validación: {len(val):,} | Test: {len(test):,}\n")
say("PASO 1 - Candidatos entrenados con train y medidos en validación (2022-2023)")
yv = val["target"].to_numpy()
val_preds, rows = {}, []
for c in candidates:
    c.fit(train)
    p = c.predict(val)
    val_preds[c.name] = p
    rows.append((c.name, acc(yv, p), logloss(yv, p)))
    say(f"  {c.name:38s} acierto={acc(yv, p):.4f}  log-loss={logloss(yv, p):.4f}")

# el conjunto: media de la regresión y de los árboles
tree_names = [c.name for c in candidates if c.kind == "hgb"]
ens = np.mean([val_preds["LogReg"], np.mean([val_preds[n] for n in tree_names], axis=0)], axis=0)
rows.append(("Conjunto (LogReg + media de árboles)", acc(yv, ens), logloss(yv, ens)))
say(f"  {'Conjunto (LogReg + media de árboles)':38s} acierto={acc(yv, ens):.4f}  log-loss={logloss(yv, ens):.4f}")

best_name = max(rows, key=lambda r: r[1])[0]
se = 100 * np.sqrt(0.65 * 0.35 / len(val))
say(f"\n  Elegido por mayor acierto en validación: {best_name}")
say(f"  Ojo: el acierto en validación tiene un margen de error de ±{se:.2f} puntos. Diferencias menores")
say("  que eso entre candidatos son ruido, no mejoras reales.")

# ---------- 2) Reentrenar con train + val y examinar en test (una sola vez) ----------
say("\nPASO 2 - Reentrenar con train + val (2005-2023) y examinar en test (2024 en adelante)")
trainval = pd.concat([train, val])
final = []
if best_name.startswith("Conjunto"):
    final = [c.fit(trainval) for c in candidates if c.name == "LogReg" or c.kind == "hgb"]
    p_logreg = final[0].predict(test)
    p_trees = np.mean([c.predict(test) for c in final[1:]], axis=0)
    pt = np.mean([p_logreg, p_trees], axis=0)
else:
    c = next(c for c in candidates if c.name == best_name).fit(trainval)
    pt = c.predict(test)

yt = test["target"].to_numpy()
say(f"  TEST: acierto={acc(yt, pt):.4f}  log-loss={logloss(yt, pt):.4f}  Brier={((pt - yt) ** 2).mean():.4f}")
say(f"  Margen de error del acierto en test: ±{100 * np.sqrt(0.65 * 0.35 / len(test)):.2f} puntos")

# ---------- 3) Acierto según lo seguro que está el modelo ----------
conf = np.maximum(pt, 1 - pt)
hit = (pt > 0.5) == (yt == 1)
say("\nPASO 3 - Acierto según la confianza del modelo (test)")
say(f"  {'confianza':>10s} {'partidos':>9s} {'% del total':>12s} {'acierto':>9s}")
for th in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    mk = conf >= th
    say(f"  {'>= ' + format(th, '.2f'):>10s} {mk.sum():9d} {mk.mean() * 100:11.1f}% {hit[mk].mean() * 100:8.1f}%")

test["prob_A"] = pt.round(4)
test["confidence"] = conf.round(4)
test["hit"] = hit.astype(int)
test["pred_winner"] = np.where(pt > 0.5, test["player_A_name"], test["player_B_name"])
test["actual_winner"] = np.where(yt == 1, test["player_A_name"], test["player_B_name"])
grp = np.where(test["level_source"] == "main", "Circuito principal", "Challenger")
say("\nAcierto por nivel (test):")
for g in ["Circuito principal", "Challenger"]:
    mk = grp == g
    say(f"  {g:20s} partidos={mk.sum():6d}  acierto={hit[mk].mean() * 100:.1f}%")

test[["match_date", "tourney_name", "level_source", "surface", "round", "player_A_name", "player_B_name",
      "prob_A", "pred_winner", "actual_winner", "hit", "confidence"]].to_csv("predictions_best.csv", index=False)
say("\nGuardado: predictions_best.csv y train_best_report.txt")
open("train_best_report.txt", "w", encoding="utf-8").write("\n".join(lines))