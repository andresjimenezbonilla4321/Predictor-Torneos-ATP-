"""
Añade características de HISTORIAL a cada partido: forma reciente, descanso,
inactividad, cansancio y cara a cara (head-to-head).

Entrada : atp_matches_elo.csv
Salida  : atp_matches_features.csv
          state/history.pkl  (estado final, para predecir partidos futuros)

Ejecutar:  python add_features.py

Igual que con el Elo: para cada partido se guarda primero lo que se sabía ANTES de
jugarlo y solo después se actualiza el historial con el resultado (sin fuga de datos).

Limitación de los datos: todos los partidos de un torneo tienen la misma fecha
(la de inicio), así que el descanso se mide entre torneos, no entre partidos.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from common import FORM_WINDOW, LAYOFF_CAP, history_state, save_state, to_day, total_games

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_elo.csv"
OUTPUT = "atp_matches_features.csv"
KEEP_DAYS = max(60, LAYOFF_CAP + 1)   # historial de partidos que se guarda por jugador
# -----------------------------------------------------------------


def main():
    m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
    assert m["tourney_date"].is_monotonic_increasing, "El archivo debe estar ordenado por fecha"

    day = to_day(m["tourney_date"].to_numpy())
    games = m["score"].apply(total_games).to_numpy()
    W, L = m["winner_id"].to_numpy(), m["loser_id"].to_numpy()
    n = len(m)

    recent = defaultdict(deque)                       # jugador -> [(día, juegos)] del último año
    results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))   # jugador -> últimos resultados (1/0)
    last_day = {}                                     # jugador -> último día jugado (sin límite)
    h2h = defaultdict(int)                            # (a, b) -> victorias de a sobre b

    cols = ["form_w", "form_l", "rest_w", "rest_l", "layoff_w", "layoff_l",
            "games14_w", "games14_l", "h2h_w", "h2h_l"]
    out = {c: np.zeros(n) for c in cols}

    def state(p, d):
        dq = recent[p]
        while dq and d - dq[0][0] > KEEP_DAYS:
            dq.popleft()
        form, rest, layoff, g14 = history_state(dq, results[p], d)
        # si su último partido es anterior a la ventana guardada, la inactividad sigue siendo correcta
        if not dq and p in last_day:
            layoff = min(d - last_day[p], LAYOFF_CAP)
        return form, rest, layoff, g14

    for i in range(n):
        w, l, d = W[i], L[i], day[i]
        # ---- 1) Estado ANTES del partido ----
        out["form_w"][i], out["rest_w"][i], out["layoff_w"][i], out["games14_w"][i] = state(w, d)
        out["form_l"][i], out["rest_l"][i], out["layoff_l"][i], out["games14_l"][i] = state(l, d)
        out["h2h_w"][i], out["h2h_l"][i] = h2h[(w, l)], h2h[(l, w)]
        # ---- 2) Actualizar con el resultado ----
        recent[w].append((d, games[i]))
        recent[l].append((d, games[i]))
        results[w].append(1)
        results[l].append(0)
        last_day[w] = last_day[l] = d
        h2h[(w, l)] += 1

    for c in cols:
        m[c] = out[c]

    m.to_csv(OUTPUT, index=False)
    # estado final (dicts normales para poder guardarlos con pickle)
    save_state("history", {
        "recent": {p: list(dq) for p, dq in recent.items()},
        "results": {p: list(dq) for p, dq in results.items()},
        "last_day": last_day,
        "h2h": dict(h2h),
    })
    print(f"Partidos procesados: {n:,}")
    print(m[cols].describe().loc[["mean", "min", "max"]].round(2).T.to_string())
    print(f"Archivo guardado: {OUTPUT}")


if __name__ == "__main__":
    main()
