"""
Añade estadísticas de SAQUE y RESTO de cada jugador, calculadas solo con sus
partidos ANTERIORES (sin fuga de datos).

Entrada : atp_matches_features.csv   (salida de add_features.py)
Salida  : atp_matches_serve.csv

Ejecutar:  python add_serve_stats.py

Por jugador se mantiene una ventana con sus últimos N partidos con estadísticas:
  spw  = % de puntos ganados al saque
  rpw  = % de puntos ganados al resto
  ace  = aces por punto de saque
  df   = dobles faltas por punto de saque
  fin  = % de primeros saques metidos
Con pocos puntos, cada valor se acerca a la media del circuito (suavizado), para
no fiarse de un jugador del que solo se conoce un partido.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_features.csv"
OUTPUT = "atp_matches_serve.csv"

WINDOW = 20        # últimos N partidos con estadísticas
SHRINK_PTS = 300   # "puntos imaginarios" con la media del circuito (~4 partidos)

# medias del circuito medidas en tus datos (todos los niveles, partidos con estadísticas)
PRIOR = {"spw": 0.622, "rpw": 0.378, "ace": 0.068, "df": 0.039, "fin": 0.610}
# -----------------------------------------------------------------

m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
assert m["tourney_date"].is_monotonic_increasing, "El archivo debe estar ordenado por fecha"
n = len(m)

W, L = m["winner_id"].to_numpy(), m["loser_id"].to_numpy()
has = (m["has_stats"] == 1).to_numpy()

num = lambda c: m[c].to_numpy(dtype=float)
w_sv, l_sv = num("w_svpt"), num("l_svpt")
w_won, l_won = num("w_1stWon") + num("w_2ndWon"), num("l_1stWon") + num("l_2ndWon")
w_ace, l_ace, w_df, l_df = num("w_ace"), num("l_ace"), num("w_df"), num("l_df")
w_in, l_in = num("w_1stIn"), num("l_1stIn")

# por jugador: ventana de tuplas (svpt, sv_ganados, pts_resto, resto_ganados, aces, dobles, primeros_in)
hist = defaultdict(lambda: deque(maxlen=WINDOW))
KEYS = ["spw", "rpw", "ace", "df", "fin", "ns"]
out = {f"{k}_{s}": np.full(n, np.nan) for k in KEYS for s in ("w", "l")}


def state(p):
    dq = hist[p]
    if not dq:
        return [PRIOR["spw"], PRIOR["rpw"], PRIOR["ace"], PRIOR["df"], PRIOR["fin"], 0]
    a = np.array(dq).sum(axis=0)
    sv, won, rt, rwon, ace, df, fin = a
    k = SHRINK_PTS
    return [
        (won + k * PRIOR["spw"]) / (sv + k),
        (rwon + k * PRIOR["rpw"]) / (rt + k),
        (ace + k * PRIOR["ace"]) / (sv + k),
        (df + k * PRIOR["df"]) / (sv + k),
        (fin + k * PRIOR["fin"]) / (sv + k),
        len(dq),
    ]


for i in range(n):
    # ---- 1) Estado ANTES del partido ----
    for side, p in (("w", W[i]), ("l", L[i])):
        for key, val in zip(KEYS, state(p)):
            out[f"{key}_{side}"][i] = val
    # ---- 2) Actualizar con este partido (solo si tiene estadísticas) ----
    if has[i]:
        hist[W[i]].append((w_sv[i], w_won[i], l_sv[i], l_sv[i] - l_won[i], w_ace[i], w_df[i], w_in[i]))
        hist[L[i]].append((l_sv[i], l_won[i], w_sv[i], w_sv[i] - w_won[i], l_ace[i], l_df[i], l_in[i]))

for c, v in out.items():
    m[c] = v

m.to_csv(OUTPUT, index=False)
print(f"Partidos procesados: {n:,}")
print(m[[c for c in out if c.endswith("_w")]].describe().loc[["mean", "min", "max"]].round(3).T.to_string())
print(f"Archivo guardado: {OUTPUT}")