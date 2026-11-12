"""
Compara el modelo con las CASAS DE APUESTAS en los partidos de test (2024 en adelante).

Fuente de cuotas: conjunto público 'dissfya/atp-tennis-2000-2023daily-pull' (Kaggle), que
recoge resultados y cuotas del circuito principal ATP (ATP 250, 500, Masters 1000, Grand
Slam y Finales; no incluye Challengers). Se descarga automáticamente en odds/.

Qué mide:
  1. Acierto, log-loss y Brier en los MISMOS partidos: casa de apuestas vs modelo vs Elo.
     La probabilidad de la casa se obtiene de las cuotas quitando su margen.
  2. Si mezclar modelo y casa mejora a la casa (peso elegido en validación 2022-2023).
  3. Apuestas simuladas: apostar 1 unidad cuando el modelo ve "valor" (su probabilidad x
     cuota > 1 + umbral) y ver el beneficio. Es un ejercicio, NO una recomendación de apuesta:
     las cuotas son de cierre y no hay garantía de que se pudiera apostar a ese precio.

Modelos usados (sin fuga de datos):
  - test (2024+): models/model_trainval.joblib (entrenado y calibrado con 2005-2023)
  - val (2022-2023, solo para elegir el peso de la mezcla): models/model_train.joblib (2005-2021)

Salidas: odds_report.txt, odds_matches.csv
Ejecutar: python compare_odds.py
"""
import io
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import MODELS_DIR, TRAIN_END_YEAR, VAL_END_YEAR, CalibratedModel, accuracy, brier, logloss

ODDS_URL = "https://www.kaggle.com/api/v1/datasets/download/dissfya/atp-tennis-2000-2023daily-pull"
ODDS_DIR = Path("odds")
ODDS_CSV = ODDS_DIR / "atp_tennis.csv"
MAX_DAYS_AFTER_START = 16      # un partido se juega como mucho ~2 semanas después del inicio del torneo
EDGES = [0.0, 0.03, 0.05, 0.10]


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return "".join(c for c in text if c.isalpha())


def odds_key(name):
    """'De Minaur A.' -> 'deminaur_a' ; 'Auger-Aliassime F.' -> 'augeraliassime_f'."""
    name = str(name).strip()
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].endswith("."):
        last, first = parts[0], parts[1]
        return f"{normalize(last)}_{normalize(first)[:1]}"
    return normalize(name)


def player_key(first, last):
    return f"{normalize(last)}_{normalize(first)[:1]}"


def load_odds():
    if not ODDS_CSV.exists():
        ODDS_DIR.mkdir(exist_ok=True)
        print(f"Descargando cuotas de {ODDS_URL} ...")
        with urllib.request.urlopen(ODDS_URL, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            ODDS_CSV.write_bytes(z.read(name))
    o = pd.read_csv(ODDS_CSV)
    o["Date"] = pd.to_datetime(o["Date"])
    o = o[(o["Odd_1"] > 1) & (o["Odd_2"] > 1)].copy()
    o["k1"], o["k2"] = o["Player_1"].map(odds_key), o["Player_2"].map(odds_key)
    # probabilidad implícita sin el margen de la casa
    inv1, inv2 = 1 / o["Odd_1"], 1 / o["Odd_2"]
    o["p1_book"] = inv1 / (inv1 + inv2)
    o["margin"] = inv1 + inv2 - 1
    o["p1_won"] = (o["Winner"] == o["Player_1"]).astype(int)
    return o


def model_predictions(split):
    """Predicciones del modelo (sin fuga) para los partidos de val o test de model_dataset.csv."""
    d = pd.read_csv("model_dataset.csv", parse_dates=["match_date"])
    d = d[d["split"] == split].copy()
    name = "model_trainval.joblib" if split == "test" else "model_train.joblib"
    bundle = joblib.load(MODELS_DIR / name)
    model = bundle["model"]
    if split == "val":
        # el modelo de solo train no está calibrado: se usa la misma calibración que el final
        a = joblib.load(MODELS_DIR / "model_trainval.joblib")["model"].a
        model = CalibratedModel(model, a)
    d["p_model"] = model.predict(d[bundle["features"]])
    return d


def match_with_odds(d, o, players):
    """Empareja cada partido nuestro con el de las cuotas (mismos jugadores, fecha dentro del torneo)."""
    key = {pid: player_key(f, l) for pid, f, l in
           zip(players["player_id"], players["name_first"], players["name_last"])}
    d = d.assign(kA=d["player_A_id"].map(key), kB=d["player_B_id"].map(key))
    d["pair"] = [frozenset((a, b)) for a, b in zip(d["kA"], d["kB"])]
    o = o.assign(pair=[frozenset((a, b)) for a, b in zip(o["k1"], o["k2"])])
    cand = d.reset_index().merge(o.reset_index().rename(columns={"index": "oidx"}), on="pair")
    delta = (cand["Date"] - cand["match_date"]).dt.days
    cand = cand[(delta >= 0) & (delta <= MAX_DAYS_AFTER_START)]
    # partidos que encajan con una única fila en cada lado
    cand = cand[~cand.duplicated("index", keep=False) & ~cand.duplicated("oidx", keep=False)]
    # probabilidad de la casa para el jugador A
    a_is_1 = cand["kA"] == cand["k1"]
    cand["p_book"] = np.where(a_is_1, cand["p1_book"], 1 - cand["p1_book"])
    cand["odd_A"] = np.where(a_is_1, cand["Odd_1"], cand["Odd_2"])
    cand["odd_B"] = np.where(a_is_1, cand["Odd_2"], cand["Odd_1"])
    # comprobación: el ganador según las cuotas debe coincidir con el nuestro
    won_a_book = np.where(a_is_1, cand["p1_won"], 1 - cand["p1_won"])
    agree = (won_a_book == cand["target"]).mean()
    return cand[won_a_book == cand["target"]].copy(), agree


def blend(p_model, p_book, w):
    """Mezcla en escala logit: w = peso del modelo."""
    lm = np.log(p_model / (1 - p_model))
    lb = np.log(p_book / (1 - p_book))
    return 1 / (1 + np.exp(-(w * lm + (1 - w) * lb)))


def main():
    lines = []

    def say(t=""):
        print(t)
        lines.append(str(t))

    o = load_odds()
    players = pd.read_csv("atp_players.csv", low_memory=False)

    test, agree_t = match_with_odds(model_predictions("test"), o, players)
    val, agree_v = match_with_odds(model_predictions("val"), o, players)
    n_main = (model_predictions("test")["level_source"] == "main").sum()

    say("COMPARACIÓN CON LAS CASAS DE APUESTAS - partidos de test (2024 en adelante)")
    say(f"Partidos emparejados con cuotas: {len(test):,} de {n_main:,} del circuito principal en test "
        f"(coincidencia de ganador: {agree_t * 100:.2f} %)")
    say(f"Margen medio de la casa: {test['margin'].mean() * 100:.1f} % (se quita para obtener su probabilidad)\n")

    y = test["target"].to_numpy()
    rows = [("Casa de apuestas", test["p_book"].to_numpy()),
            ("Modelo", test["p_model"].to_numpy()),
            ("Elo (fórmula)", test["elo_prob_A"].to_numpy())]

    # ---------- mezcla modelo + casa: peso elegido en VALIDACIÓN ----------
    yv = val["target"].to_numpy()
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    lls = {w: logloss(yv, blend(val["p_model"].to_numpy(), val["p_book"].to_numpy(), w)) for w in grid}
    w_best = float(min(lls, key=lls.get))
    rows.append((f"Mezcla (peso modelo {w_best:.2f})",
                 blend(test["p_model"].to_numpy(), test["p_book"].to_numpy(), w_best)))

    say("1) MISMOS PARTIDOS, TRES PREDICTORES")
    say(f"   {'':30s} {'acierto':>8s} {'log-loss':>9s} {'Brier':>7s}")
    for name, p in rows:
        say(f"   {name:30s} {accuracy(y, p) * 100:7.2f}% {logloss(y, p):9.4f} {brier(y, p):7.4f}")
    say(f"\n   Peso de la mezcla elegido en validación {TRAIN_END_YEAR + 1}-{VAL_END_YEAR} "
        f"({len(val):,} partidos con cuotas): {w_best:.2f}")

    # diferencia de log-loss modelo - casa con IC 95 %
    pb, pm = np.clip(test["p_book"], 1e-6, 1 - 1e-6), np.clip(test["p_model"], 1e-6, 1 - 1e-6)
    ll_b = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    ll_m = -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    diff = ll_m - ll_b
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    say(f"   Diferencia de log-loss modelo - casa: {diff.mean():+.4f} [{diff.mean() - half:+.4f}, "
        f"{diff.mean() + half:+.4f}] IC 95 % (> 0 = la casa es mejor)")

    say("\n   Por categoría (acierto modelo / casa):")
    for s, g in test.groupby("Series"):
        yy = g["target"].to_numpy()
        say(f"   {s:14s} {len(g):5d} partidos   modelo {accuracy(yy, g['p_model']) * 100:5.1f} %   "
            f"casa {accuracy(yy, g['p_book']) * 100:5.1f} %")

    # ---------- 2) ¿Coinciden? ----------
    same = (test["p_model"] > 0.5) == (test["p_book"] > 0.5)
    say(f"\n2) EL MODELO Y LA CASA ELIGEN AL MISMO FAVORITO en el {same.mean() * 100:.1f} % de los partidos.")
    dis = test[~same]
    say(f"   Cuando no coinciden ({len(dis):,} partidos): acierta el modelo el "
        f"{accuracy(dis['target'], dis['p_model']) * 100:.1f} %, la casa el "
        f"{accuracy(dis['target'], dis['p_book']) * 100:.1f} %")

    # ---------- 3) Apuestas simuladas ----------
    say("\n3) APUESTAS SIMULADAS (1 unidad por apuesta, cuotas de la fuente) - solo como ejercicio")
    say("   Se apuesta al jugador cuando prob_modelo x cuota > 1 + umbral.")
    say(f"   {'umbral':>7s} {'apuestas':>9s} {'acertadas':>10s} {'beneficio':>10s} {'ROI':>8s}")
    pA, pB = test["p_model"].to_numpy(), 1 - test["p_model"].to_numpy()
    oA, oB = test["odd_A"].to_numpy(), test["odd_B"].to_numpy()
    for e in EDGES:
        betA = pA * oA > 1 + e
        betB = (pB * oB > 1 + e) & ~betA
        stake = betA.sum() + betB.sum()
        ret = (betA * (y == 1) * oA).sum() + (betB * (y == 0) * oB).sum()
        wins = (betA & (y == 1)).sum() + (betB & (y == 0)).sum()
        roi = (ret - stake) / stake * 100 if stake else 0
        say(f"   {e * 100:6.0f}% {stake:9d} {wins:10d} {ret - stake:+10.1f} {roi:+7.1f}%")
    say("   (ROI negativo = se pierde dinero. Superar a las casas de forma sostenida es muy difícil.)")

    test[["match_date", "tourney_name", "Series", "round", "player_A_name", "player_B_name", "p_model",
          "p_book", "elo_prob_A", "odd_A", "odd_B", "target"]].to_csv("odds_matches.csv", index=False)
    say("\nGuardado: odds_report.txt y odds_matches.csv")
    with open("odds_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
