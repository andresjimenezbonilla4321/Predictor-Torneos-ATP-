"""
Añade estadísticas de SAQUE y RESTO de cada jugador, calculadas solo con sus
partidos ANTERIORES (sin fuga de datos).

Entrada : atp_matches_features.csv   (salida de add_features.py)
Salida  : atp_matches_serve.csv
          state/serve.pkl  (estado final, para predecir partidos futuros)

Ejecutar:  python add_serve_stats.py

Por jugador se mantiene una ventana con sus últimos N partidos con estadísticas:
  spw  = % de puntos ganados al saque
  rpw  = % de puntos ganados al resto
  ace  = aces por punto de saque
  df   = dobles faltas por punto de saque
  fin  = % de primeros saques metidos
Con pocos puntos, cada valor se acerca a la media del circuito (suavizado), para
no fiarse de un jugador del que solo se conoce un partido.

Las medias del circuito se calculan SOLO con los años de entrenamiento, para que
validación y test no influyan en nada de lo que ve el modelo.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from common import SERVE_WINDOW, TRAIN_END_YEAR, save_state, serve_state

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_features.csv"
OUTPUT = "atp_matches_serve.csv"
# -----------------------------------------------------------------


def circuit_prior(m):
    """Medias del circuito con los partidos con estadísticas hasta TRAIN_END_YEAR."""
    t = m[(m["has_stats"] == 1) & (m["tourney_date"].dt.year <= TRAIN_END_YEAR)]
    sv = t["w_svpt"].sum() + t["l_svpt"].sum()
    won = (t["w_1stWon"] + t["w_2ndWon"] + t["l_1stWon"] + t["l_2ndWon"]).sum()
    spw = won / sv
    return {
        "spw": float(spw),
        "rpw": float(1 - spw),
        "ace": float((t["w_ace"].sum() + t["l_ace"].sum()) / sv),
        "df": float((t["w_df"].sum() + t["l_df"].sum()) / sv),
        "fin": float((t["w_1stIn"].sum() + t["l_1stIn"].sum()) / sv),
    }


def main():
    m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
    assert m["tourney_date"].is_monotonic_increasing, "El archivo debe estar ordenado por fecha"
    n = len(m)

    prior = circuit_prior(m)
    print("Medias del circuito (años de train):", {k: round(v, 3) for k, v in prior.items()})

    W, L = m["winner_id"].to_numpy(), m["loser_id"].to_numpy()
    has = (m["has_stats"] == 1).to_numpy()

    num = lambda c: m[c].to_numpy(dtype=float)
    w_sv, l_sv = num("w_svpt"), num("l_svpt")
    w_won, l_won = num("w_1stWon") + num("w_2ndWon"), num("l_1stWon") + num("l_2ndWon")
    w_ace, l_ace, w_df, l_df = num("w_ace"), num("l_ace"), num("w_df"), num("l_df")
    w_in, l_in = num("w_1stIn"), num("l_1stIn")

    # por jugador: ventana de tuplas (svpt, sv_ganados, pts_resto, resto_ganados, aces, dobles, primeros_in)
    hist = defaultdict(lambda: deque(maxlen=SERVE_WINDOW))
    KEYS = ["spw", "rpw", "ace", "df", "fin", "ns"]
    out = {f"{k}_{s}": np.full(n, np.nan) for k in KEYS for s in ("w", "l")}

    for i in range(n):
        # ---- 1) Estado ANTES del partido ----
        for side, p in (("w", W[i]), ("l", L[i])):
            for key, val in zip(KEYS, serve_state(hist[p], prior)):
                out[f"{key}_{side}"][i] = val
        # ---- 2) Actualizar con este partido (solo si tiene estadísticas) ----
        if has[i]:
            hist[W[i]].append((w_sv[i], w_won[i], l_sv[i], l_sv[i] - l_won[i], w_ace[i], w_df[i], w_in[i]))
            hist[L[i]].append((l_sv[i], l_won[i], w_sv[i], w_sv[i] - w_won[i], l_ace[i], l_df[i], l_in[i]))

    for c, v in out.items():
        m[c] = v

    m.to_csv(OUTPUT, index=False)
    save_state("serve", {"hist": {p: list(dq) for p, dq in hist.items()}, "prior": prior})
    print(f"Partidos procesados: {n:,}")
    print(m[[c for c in out if c.endswith("_w")]].describe().loc[["mean", "min", "max"]].round(3).T.to_string())
    print(f"Archivo guardado: {OUTPUT}")


if __name__ == "__main__":
    main()
