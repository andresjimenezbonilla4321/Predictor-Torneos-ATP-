"""
Calcula el Elo general y por superficie de cada jugador.

Entrada : atp_matches_clean.csv
Salida  : atp_matches_elo.csv     (cada partido con el Elo de ambos ANTES de jugarlo)
          elo_ratings_final.csv   (valoración actual de cada jugador)
          state/elo.pkl           (estado final, para predecir partidos futuros)

Ejecutar:  python compute_elo.py
           python compute_elo.py --tune   (ajusta los parámetros mirando SOLO 2005-2021)

IMPORTANTE (sin fuga de datos): para cada partido se guarda primero el Elo que
tenían los jugadores y solo DESPUÉS se actualiza con el resultado.

Mejoras sobre el Elo clásico:
  - K variable: alto para jugadores nuevos, baja con la experiencia.
  - Margen de victoria: un 6-1 6-1 mueve más el Elo que un 7-6 6-7 7-6.
  - Partidos al mejor de 5 sets: más informativos (menos azar), pueden pesar más.
  - Retiradas: pesan menos.
  - Inactividad: tras una ausencia larga, el Elo se acerca a 1500.
"""
import sys
import time
from bisect import bisect_right

import numpy as np
import pandas as pd

from common import (FIRST_YEAR, START_RATING, TRAIN_END_YEAR, accuracy, brier, expected, games_split,
                    inactivity_shrink, logloss, save_state, to_day)

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_clean.csv"
OUTPUT = "atp_matches_elo.csv"
FINAL = "elo_ratings_final.csv"

# Parámetros elegidos con --tune (log-loss en 2005-2021 de la media Elo general + superficie).
# Ver el informe elo_tuning.txt.
# Antes del ajuste (K 250/5/0.4, sin margen de victoria): 0.6196 | tras el ajuste: 0.6165.
PARAMS = {
    "k_base": 200.0,      # K = k_base / (partidos + k_offset) ** k_shape
    "k_offset": 20.0,
    "k_shape": 0.4,
    "k_surf_mult": 1.0,   # multiplicador de K para el Elo de superficie
    "ret_weight": 0.75,   # peso de un partido terminado por retirada
    "mov": 1.5,           # margen de victoria: K se multiplica por (1 + margen) ** mov
    "bo5_mult": 1.15,     # multiplicador de K en partidos al mejor de 5 sets
    "shrink": 0.30,       # inactividad: fracción de la distancia a 1500 que se pierde en un año fuera
    "min_days": 120,      # ... a partir de estos días sin jugar
    "surf_init": False,   # primer partido en una superficie: empezar desde el Elo general
}
TREND_DAYS = 90           # tendencia = Elo actual - Elo de hace TREND_DAYS días
# -----------------------------------------------------------------


def run_elo(m, p=None, arrays=None):
    """Recorre los partidos en orden y devuelve (columnas pre-partido, estado final)."""
    p = {**PARAMS, **(p or {})}
    n = len(m)
    if arrays is None:
        arrays = match_arrays(m)
    W, L, S, RET, D, MARGIN, BO5 = arrays

    gen_r, gen_n = {}, {}      # Elo general y nº de partidos por jugador
    srf_r, srf_n = {}, {}      # Elo por superficie: clave (jugador, superficie)
    last_day = {}              # último día en que jugó cada jugador
    hist_d, hist_r = {}, {}    # historial (día, Elo tras el partido) de cada jugador, para la tendencia

    cols = ["elo_w", "elo_l", "n_w", "n_l", "elo_surf_w", "elo_surf_l", "n_surf_w", "n_surf_l",
            "trend_w", "trend_l"]
    out = {c: np.full(n, np.nan) for c in cols}
    kb, ko, ks = p["k_base"], p["k_offset"], p["k_shape"]
    shrink, min_days = p["shrink"], p["min_days"]

    for i in range(n):
        w, l, s, d = W[i], L[i], S[i], D[i]
        weight = (p["ret_weight"] if RET[i] else 1.0) * ((1.0 + MARGIN[i]) ** p["mov"]) \
            * (p["bo5_mult"] if BO5[i] else 1.0)
        has_surface = isinstance(s, str)

        # ---- 0) Ajuste por inactividad (solo usa fechas pasadas) ----
        for pl in (w, l):
            if pl in last_day and d - last_day[pl] > min_days:
                gap = d - last_day[pl]
                gen_r[pl] = inactivity_shrink(gen_r[pl], gap, min_days, shrink)
                for surf in ("Hard", "Clay", "Grass", "Carpet"):
                    if (pl, surf) in srf_r:
                        srf_r[(pl, surf)] = inactivity_shrink(srf_r[(pl, surf)], gap, min_days, shrink)
            last_day[pl] = d

        # ---- 1) Estado ANTES del partido (esto es lo que usará el modelo) ----
        rw, rl = gen_r.get(w, START_RATING), gen_r.get(l, START_RATING)
        nw, nl = gen_n.get(w, 0), gen_n.get(l, 0)
        out["elo_w"][i], out["elo_l"][i] = rw, rl
        out["n_w"][i], out["n_l"][i] = nw, nl
        out["trend_w"][i] = elo_trend(hist_d, hist_r, w, rw, d)
        out["trend_l"][i] = elo_trend(hist_d, hist_r, l, rl, d)

        if has_surface:
            sw = srf_r.get((w, s), rw if p["surf_init"] else START_RATING)
            sl = srf_r.get((l, s), rl if p["surf_init"] else START_RATING)
            snw, snl = srf_n.get((w, s), 0), srf_n.get((l, s), 0)
            out["elo_surf_w"][i], out["elo_surf_l"][i] = sw, sl
            out["n_surf_w"][i], out["n_surf_l"][i] = snw, snl

        # ---- 2) Actualizar con el resultado (el ganador es siempre "w") ----
        surprise = 1.0 - 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))   # cuanto más inesperado, más se mueve
        gen_r[w] = rw + kb / (nw + ko) ** ks * weight * surprise
        gen_r[l] = rl - kb / (nl + ko) ** ks * weight * surprise
        gen_n[w], gen_n[l] = nw + 1, nl + 1
        for pl in (w, l):
            hist_d.setdefault(pl, []).append(d)
            hist_r.setdefault(pl, []).append(gen_r[pl])

        if has_surface:
            surprise_s = 1.0 - 1.0 / (1.0 + 10 ** ((sl - sw) / 400.0))
            ksm = p["k_surf_mult"]
            srf_r[(w, s)] = sw + ksm * kb / (snw + ko) ** ks * weight * surprise_s
            srf_r[(l, s)] = sl - ksm * kb / (snl + ko) ** ks * weight * surprise_s
            srf_n[(w, s)], srf_n[(l, s)] = snw + 1, snl + 1

    state = {"gen_r": gen_r, "gen_n": gen_n, "srf_r": srf_r, "srf_n": srf_n, "last_day": last_day,
             "hist_d": hist_d, "hist_r": hist_r, "trend_days": TREND_DAYS,
             "params": dict(p), "last_date": str(m["tourney_date"].max().date())}
    return out, state


def elo_trend(hist_d, hist_r, p, r_now, d, days=TREND_DAYS):
    """Elo actual menos el Elo que tenía hace 'days' días (0 si no hay historial tan antiguo)."""
    ds = hist_d.get(p)
    if not ds:
        return 0.0
    j = bisect_right(ds, d - days)
    return float(r_now - hist_r[p][j - 1]) if j > 0 else 0.0


def match_arrays(m):
    """Columnas que necesita el bucle del Elo, precalculadas una vez."""
    g = np.array([games_split(s) for s in m["score"].to_numpy()], dtype=float)
    total = g.sum(axis=1)
    # margen = (juegos del ganador - juegos del perdedor) / juegos totales, en [0, 1]
    margin = np.where(total > 0, (g[:, 0] - g[:, 1]) / np.maximum(total, 1), 0.0).clip(0, 1)
    return (m["winner_id"].to_numpy(), m["loser_id"].to_numpy(), m["surface"].to_numpy(),
            (m["is_retirement"] == 1).to_numpy(), to_day(m["tourney_date"].to_numpy()), margin,
            (m["best_of"] == 5).to_numpy())


def evaluate(m, out, years):
    """Log-loss del Elo (general, superficie y media de ambos) en los años indicados."""
    yr = m["tourney_date"].dt.year.to_numpy()
    mk = (yr >= years[0]) & (yr <= years[1])
    p = expected(out["elo_w"][mk], out["elo_l"][mk])
    ps = expected(out["elo_surf_w"][mk], out["elo_surf_l"][mk])
    ok = ~np.isnan(ps)
    ones = np.ones(mk.sum())
    return logloss(ones, p), logloss(ones[ok], ps[ok]), logloss(ones[ok], 0.5 * p[ok] + 0.5 * ps[ok])


# valores que prueba --tune para cada parámetro (búsqueda por coordenadas)
GRID = {
    "k_base": [150, 200, 250, 300, 350, 450],
    "k_offset": [2, 5, 10, 20],
    "k_shape": [0.3, 0.35, 0.4, 0.45, 0.5],
    "k_surf_mult": [0.8, 1.0, 1.2, 1.4],
    "ret_weight": [0.25, 0.5, 0.75],
    "mov": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "bo5_mult": [1.0, 1.15, 1.3, 1.5],
    "shrink": [0.1, 0.2, 0.3],
    "min_days": [60, 120, 180],
}


def tune(m, rounds=2):
    """
    Busca los parámetros que minimizan el log-loss de la media Elo general + superficie,
    mirando SOLO 2005-2021 (validación y test no influyen). Se ajusta un parámetro cada vez.
    """
    arrays = match_arrays(m)
    years = (FIRST_YEAR, TRAIN_END_YEAR)
    lines = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    cache = {}

    def score(p):
        key = tuple(sorted(p.items()))
        if key not in cache:
            out, _ = run_elo(m, p, arrays)
            cache[key] = evaluate(m, out, years)
        return cache[key]

    best = dict(PARAMS)
    t0 = time.time()
    g0, s0, mix0 = score(best)
    say(f"Ajuste del Elo con log-loss en {years[0]}-{years[1]} (menor = mejor). Solo años de entrenamiento.")
    say(f"Punto de partida: general {g0:.4f} | superficie {s0:.4f} | media {mix0:.4f}\n")
    for r in range(rounds):
        say(f"Ronda {r + 1}")
        for name, values in GRID.items():
            results = {v: score({**best, name: v})[2] for v in values}
            v_best = min(results, key=results.get)
            txt = "  ".join(f"{v}:{results[v]:.4f}" for v in values)
            changed = "  <- cambia" if v_best != best[name] else ""
            say(f"  {name:12s} {txt}   -> {v_best}{changed}")
            best[name] = v_best
    g, s, mix = score(best)
    say(f"\nFinal: general {g:.4f} | superficie {s:.4f} | media {mix:.4f}  "
        f"(mejora de la media: {mix0 - mix:+.4f})  [{time.time() - t0:.0f} s]")
    say("Parámetros: " + ", ".join(f'"{k}": {v}' for k, v in best.items()))
    with open("elo_tuning.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return best


def main():
    m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
    assert m["tourney_date"].is_monotonic_increasing, "El archivo debe estar ordenado por fecha"

    if "--tune" in sys.argv:
        tune(m)
        return

    out, state = run_elo(m)
    for c, v in out.items():
        m[c] = v
    m["elo_prob_w"] = expected(m["elo_w"], m["elo_l"])
    m["elo_surf_prob_w"] = expected(m["elo_surf_w"], m["elo_surf_l"])
    m.to_csv(OUTPUT, index=False)
    save_state("elo", state)

    gen_r, gen_n, srf_r = state["gen_r"], state["gen_n"], state["srf_r"]

    # ---------------- Ratings finales por jugador ----------------
    names = pd.concat([
        m[["winner_id", "winner_name", "tourney_date"]].set_axis(["player_id", "name", "date"], axis=1),
        m[["loser_id", "loser_name", "tourney_date"]].set_axis(["player_id", "name", "date"], axis=1),
    ]).sort_values("date").drop_duplicates("player_id", keep="last")
    final = names.assign(elo=names["player_id"].map(gen_r), n_matches=names["player_id"].map(gen_n))
    for surf in ["Hard", "Clay", "Grass"]:
        final[f"elo_{surf.lower()}"] = [srf_r.get((p, surf), np.nan) for p in final["player_id"]]
    final = final.rename(columns={"date": "last_match"})
    # Elo "a día de hoy" (fin de los datos): aplica la inactividad acumulada desde su último partido
    end = m["tourney_date"].max()
    gap = (end - final["last_match"]).dt.days
    final["elo_now"] = [inactivity_shrink(r, g, PARAMS["min_days"], PARAMS["shrink"])
                        for r, g in zip(final["elo"], gap)]
    final["active_12m"] = (gap <= 365).astype(int)
    final = final.sort_values("elo_now", ascending=False)
    final.to_csv(FINAL, index=False)

    # ---------------- Validación ----------------
    print(f"Partidos procesados: {len(m):,} | jugadores: {len(gen_r):,}")
    v = m[m["tourney_date"].dt.year >= FIRST_YEAR]

    def report(name, p):
        ones = np.ones(len(p))
        print(f"{name:34s} acierto: {accuracy(ones, p):6.1%} | log-loss: {logloss(ones, p):.4f} | "
              f"Brier: {brier(ones, p):.4f}")

    print(f"\nValidación (partidos desde {FIRST_YEAR}, la 'probabilidad del ganador' debería ser > 0.5):")
    report("Elo general", v["elo_prob_w"].to_numpy())
    vs = v.dropna(subset=["elo_surf_prob_w"])
    report("Elo de la superficie", vs["elo_surf_prob_w"].to_numpy())
    report("Media Elo general + superficie", (0.5 * vs["elo_prob_w"] + 0.5 * vs["elo_surf_prob_w"]).to_numpy())
    r = v.dropna(subset=["winner_rank", "loser_rank"])
    r = r[r["winner_rank"] != r["loser_rank"]]
    print(f"{'Mejor ranking ATP gana':34s} acierto: {(r['winner_rank'] < r['loser_rank']).mean():6.1%}")

    print("\nTop 10 Elo entre jugadores activos en los últimos 12 meses del dataset:")
    recent = final[final["active_12m"] == 1]
    print(recent.head(10)[["name", "elo_now", "elo_hard", "elo_clay", "elo_grass", "n_matches"]]
          .round(0).to_string(index=False))


if __name__ == "__main__":
    main()
