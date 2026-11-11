"""
Calcula el Elo general y por superficie de cada jugador.

Entrada : atp_matches_clean.csv
Salida  : atp_matches_elo.csv     (cada partido con el Elo de ambos ANTES de jugarlo)
          elo_ratings_final.csv   (valoración actual de cada jugador)
          state/elo.pkl           (estado final, para predecir partidos futuros)

Ejecutar:  python compute_elo.py
           python compute_elo.py --tune   (prueba parámetros y compara en 2005-2021)

IMPORTANTE (sin fuga de datos): para cada partido se guarda primero el Elo que
tenían los jugadores y solo DESPUÉS se actualiza con el resultado.
"""
import sys

import numpy as np
import pandas as pd

from common import (FIRST_YEAR, START_RATING, TRAIN_END_YEAR, accuracy, brier, expected,
                    inactivity_shrink, logloss, save_state, to_day)

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_clean.csv"
OUTPUT = "atp_matches_elo.csv"
FINAL = "elo_ratings_final.csv"

# K variable: alto para jugadores nuevos (aprenden rápido), baja con la experiencia
K_BASE, K_OFFSET, K_SHAPE = 250.0, 5.0, 0.4
# Una retirada es menos informativa que un partido completo
RETIREMENT_WEIGHT = 0.5
# Inactividad: tras más de INACTIVITY_DAYS sin jugar, el Elo se acerca a 1500
# (un año fuera -> se pierde INACTIVITY_SHRINK de la distancia a 1500). Elegido con --tune.
# Con --tune (2005-2021): sin inactividad log-loss 0.6257, con (0.20, 120 días) 0.6238.
INACTIVITY_DAYS = 120
INACTIVITY_SHRINK = 0.20
# Primer partido en una superficie: empezar desde su Elo general en vez de 1500.
# Con --tune no mejora la media general+superficie, así que se deja desactivado.
SURFACE_INIT_FROM_GENERAL = False
# -----------------------------------------------------------------


def k_factor(n_matches):
    return K_BASE / ((n_matches + K_OFFSET) ** K_SHAPE)


def run_elo(m, shrink=INACTIVITY_SHRINK, min_days=INACTIVITY_DAYS, surf_init=SURFACE_INIT_FROM_GENERAL):
    """Recorre los partidos en orden y devuelve (columnas pre-partido, estado final)."""
    n = len(m)
    gen_r, gen_n = {}, {}      # Elo general y nº de partidos por jugador
    srf_r, srf_n = {}, {}      # Elo por superficie: clave (jugador, superficie)
    last_day = {}              # último día en que jugó cada jugador

    cols = ["elo_w", "elo_l", "n_w", "n_l", "elo_surf_w", "elo_surf_l", "n_surf_w", "n_surf_l"]
    out = {c: np.full(n, np.nan) for c in cols}
    days = to_day(m["tourney_date"].to_numpy())

    for i, (w, l, s, ret, d) in enumerate(zip(m["winner_id"].values, m["loser_id"].values,
                                              m["surface"].values, m["is_retirement"].values, days)):
        weight = RETIREMENT_WEIGHT if ret == 1 else 1.0
        has_surface = isinstance(s, str)

        # ---- 0) Ajuste por inactividad (solo usa fechas pasadas) ----
        for p in (w, l):
            if p in last_day and d - last_day[p] > min_days:
                gap = d - last_day[p]
                gen_r[p] = inactivity_shrink(gen_r[p], gap, min_days, shrink)
                for surf in ("Hard", "Clay", "Grass", "Carpet"):
                    if (p, surf) in srf_r:
                        srf_r[(p, surf)] = inactivity_shrink(srf_r[(p, surf)], gap, min_days, shrink)
            last_day[p] = d

        # ---- 1) Estado ANTES del partido (esto es lo que usará el modelo) ----
        rw, rl = gen_r.get(w, START_RATING), gen_r.get(l, START_RATING)
        nw, nl = gen_n.get(w, 0), gen_n.get(l, 0)
        out["elo_w"][i], out["elo_l"][i] = rw, rl
        out["n_w"][i], out["n_l"][i] = nw, nl

        if has_surface:
            init_w = rw if surf_init else START_RATING
            init_l = rl if surf_init else START_RATING
            sw, sl = srf_r.get((w, s), init_w), srf_r.get((l, s), init_l)
            snw, snl = srf_n.get((w, s), 0), srf_n.get((l, s), 0)
            out["elo_surf_w"][i], out["elo_surf_l"][i] = sw, sl
            out["n_surf_w"][i], out["n_surf_l"][i] = snw, snl

        # ---- 2) Actualizar con el resultado (el ganador es siempre "w") ----
        surprise = 1.0 - expected(rw, rl)          # cuanto más inesperado, más se mueve
        gen_r[w] = rw + k_factor(nw) * weight * surprise
        gen_r[l] = rl - k_factor(nl) * weight * surprise
        gen_n[w], gen_n[l] = nw + 1, nl + 1

        if has_surface:
            surprise_s = 1.0 - expected(sw, sl)
            srf_r[(w, s)] = sw + k_factor(snw) * weight * surprise_s
            srf_r[(l, s)] = sl - k_factor(snl) * weight * surprise_s
            srf_n[(w, s)], srf_n[(l, s)] = snw + 1, snl + 1

    state = {"gen_r": gen_r, "gen_n": gen_n, "srf_r": srf_r, "srf_n": srf_n, "last_day": last_day,
             "params": {"shrink": shrink, "min_days": min_days, "surf_init": surf_init},
             "last_date": str(m["tourney_date"].max().date())}
    return out, state


def evaluate(m, out, years):
    """Log-loss del Elo (general y superficie) en los años indicados."""
    yr = m["tourney_date"].dt.year.to_numpy()
    mk = (yr >= years[0]) & (yr <= years[1])
    p = expected(out["elo_w"][mk], out["elo_l"][mk])
    ps = expected(out["elo_surf_w"][mk], out["elo_surf_l"][mk])
    ok = ~np.isnan(ps)
    ones = np.ones(mk.sum())
    return logloss(ones, p), logloss(ones[ok], ps[ok]), logloss(ones[ok], 0.5 * p[ok] + 0.5 * ps[ok])


def tune(m):
    """Elige los parámetros de inactividad/superficie mirando SOLO los años de entrenamiento."""
    print(f"Ajuste de parámetros con log-loss en {FIRST_YEAR}-{TRAIN_END_YEAR} (menor = mejor)\n")
    print(f"{'shrink':>7s} {'días':>5s} {'sup.init':>9s} {'general':>9s} {'superficie':>11s} {'media':>8s}")
    for surf_init in (False, True):
        for shrink in (0.0, 0.1, 0.15, 0.2, 0.3):
            for min_days in ((60,) if shrink == 0 else (60, 120)):
                out, _ = run_elo(m, shrink, min_days, surf_init)
                g, s, mix = evaluate(m, out, (FIRST_YEAR, TRAIN_END_YEAR))
                print(f"{shrink:7.2f} {min_days:5d} {str(surf_init):>9s} {g:9.4f} {s:11.4f} {mix:8.4f}")


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
    final["elo_now"] = [inactivity_shrink(r, g, INACTIVITY_DAYS, INACTIVITY_SHRINK)
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
