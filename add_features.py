"""
Añade características de HISTORIAL a cada partido: forma reciente, descanso/cansancio
y cara a cara (head-to-head).

Entrada : atp_matches_elo.csv
Salida  : atp_matches_features.csv

Ejecutar:  python add_features.py

Igual que con el Elo: para cada partido se guarda primero lo que se sabía ANTES de
jugarlo y solo después se actualiza el historial con el resultado (sin fuga de datos).
"""
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_elo.csv"
OUTPUT = "atp_matches_features.csv"

FORM_WINDOW = 10        # forma = % de victorias en los últimos N partidos
FORM_PRIOR = 5          # "partidos imaginarios" al 50% para no fiarse de 1-2 partidos
FATIGUE_DAYS = 14       # ventana para contar juegos jugados recientemente
REST_CAP = 30           # días de descanso máximos que consideramos
# -----------------------------------------------------------------


def total_games(score):
    s = re.sub(r"\[.*?\]", "", str(score))
    return sum(int(a) + int(b) for a, b in re.findall(r"(\d+)-(\d+)", s))


m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
assert m["tourney_date"].is_monotonic_increasing, "El archivo debe estar ordenado por fecha"

day = (m["tourney_date"] - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
games = m["score"].apply(total_games).to_numpy()
W, L = m["winner_id"].to_numpy(), m["loser_id"].to_numpy()
n = len(m)

recent = defaultdict(deque)                                   # jugador -> [(día, juegos)] últimos 60 días
results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))      # jugador -> últimos resultados (1/0)
h2h = defaultdict(int)                                        # (a, b) -> victorias de a sobre b

cols = ["form_w", "form_l", "rest_w", "rest_l", "games14_w", "games14_l", "h2h_w", "h2h_l"]
out = {c: np.zeros(n) for c in cols}


def state(p, d):
    dq = recent[p]
    while dq and d - dq[0][0] > 60:
        dq.popleft()
    g14 = sum(g for (dd, g) in dq if d - dd < FATIGUE_DAYS)       # incluye rondas previas del mismo torneo
    prev = [dd for (dd, g) in dq if dd < d]                        # último torneo anterior
    rest = min(d - max(prev), REST_CAP) if prev else REST_CAP
    res = results[p]
    form = (sum(res) + FORM_PRIOR * 0.5) / (len(res) + FORM_PRIOR)
    return form, rest, g14


for i in range(n):
    w, l, d = W[i], L[i], day[i]
    # ---- 1) Estado ANTES del partido ----
    out["form_w"][i], out["rest_w"][i], out["games14_w"][i] = state(w, d)
    out["form_l"][i], out["rest_l"][i], out["games14_l"][i] = state(l, d)
    out["h2h_w"][i], out["h2h_l"][i] = h2h[(w, l)], h2h[(l, w)]
    # ---- 2) Actualizar con el resultado ----
    recent[w].append((d, games[i]))
    recent[l].append((d, games[i]))
    results[w].append(1)
    results[l].append(0)
    h2h[(w, l)] += 1

for c in cols:
    m[c] = out[c]

m.to_csv(OUTPUT, index=False)
print(f"Partidos procesados: {n:,}")
print(m[cols].describe().loc[["mean", "min", "max"]].round(2).T.to_string())
print(f"Archivo guardado: {OUTPUT}")