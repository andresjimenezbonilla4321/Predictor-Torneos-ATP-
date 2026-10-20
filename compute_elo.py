"""
Calcula el Elo general y por superficie de cada jugador.

Entrada : atp_matches_clean.csv
Salida  : atp_matches_elo.csv     (cada partido con el Elo de ambos ANTES de jugarlo)
          elo_ratings_final.csv   (valoración actual de cada jugador)

Ejecutar:  python compute_elo.py

IMPORTANTE (sin fuga de datos): para cada partido se guarda primero el Elo que
tenían los jugadores y solo DESPUÉS se actualiza con el resultado.
"""
import numpy as np
import pandas as pd

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_clean.csv"
OUTPUT = "atp_matches_elo.csv"
FINAL = "elo_ratings_final.csv"

START_RATING = 1500.0
# K variable: alto para jugadores nuevos (aprenden rápido), baja con la experiencia
K_BASE, K_OFFSET, K_SHAPE = 250.0, 5.0, 0.4
# Una retirada es menos informativa que un partido completo
RETIREMENT_WEIGHT = 0.5
VALIDATE_FROM_YEAR = 2005   # los primeros años son "calentamiento" del Elo
# -----------------------------------------------------------------


def expected(r_a, r_b):
    """Probabilidad de que A gane a B según la diferencia de Elo."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def k_factor(n_matches):
    return K_BASE / ((n_matches + K_OFFSET) ** K_SHAPE)


m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
assert m["tourney_date"].is_monotonic_increasing, "El archivo debe estar ordenado por fecha"
n = len(m)

gen_r, gen_n = {}, {}      # Elo general y nº de partidos por jugador
srf_r, srf_n = {}, {}      # Elo por superficie: clave (jugador, superficie)

cols = ["elo_w", "elo_l", "n_w", "n_l", "elo_surf_w", "elo_surf_l", "n_surf_w", "n_surf_l"]
out = {c: np.full(n, np.nan) for c in cols}

for i, (w, l, s, ret) in enumerate(zip(m["winner_id"].values, m["loser_id"].values,
                                       m["surface"].values, m["is_retirement"].values)):
    weight = RETIREMENT_WEIGHT if ret == 1 else 1.0

    # ---- 1) Estado ANTES del partido (esto es lo que usará el modelo) ----
    rw, rl = gen_r.get(w, START_RATING), gen_r.get(l, START_RATING)
    nw, nl = gen_n.get(w, 0), gen_n.get(l, 0)
    out["elo_w"][i], out["elo_l"][i] = rw, rl
    out["n_w"][i], out["n_l"][i] = nw, nl

    has_surface = isinstance(s, str)
    if has_surface:
        sw, sl = srf_r.get((w, s), START_RATING), srf_r.get((l, s), START_RATING)
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

for c in cols:
    m[c] = out[c]

m["elo_prob_w"] = expected(m["elo_w"], m["elo_l"])
m["elo_surf_prob_w"] = expected(m["elo_surf_w"], m["elo_surf_l"])
m.to_csv(OUTPUT, index=False)

# ---------------- Ratings finales por jugador ----------------
names = pd.concat([
    m[["winner_id", "winner_name", "tourney_date"]].set_axis(["player_id", "name", "date"], axis=1),
    m[["loser_id", "loser_name", "tourney_date"]].set_axis(["player_id", "name", "date"], axis=1),
]).sort_values("date").drop_duplicates("player_id", keep="last")
final = names.assign(elo=names["player_id"].map(gen_r), n_matches=names["player_id"].map(gen_n))
for surf in ["Hard", "Clay", "Grass"]:
    final[f"elo_{surf.lower()}"] = [srf_r.get((p, surf), np.nan) for p in final["player_id"]]
final = final.rename(columns={"date": "last_match"}).sort_values("elo", ascending=False)
final.to_csv(FINAL, index=False)

# ---------------- Validación ----------------
print(f"Partidos procesados: {n:,} | jugadores: {len(gen_r):,}")
v = m[(m["tourney_date"].dt.year >= VALIDATE_FROM_YEAR)].copy()


def report(name, p):
    p = p.clip(1e-6, 1 - 1e-6)
    acc = ((p > 0.5) + 0.5 * (p == 0.5)).mean()
    print(f"{name:34s} acierto: {acc:6.1%} | log-loss: {-np.log(p).mean():.4f} | Brier: {((1 - p) ** 2).mean():.4f}")


print(f"\nValidación (partidos desde {VALIDATE_FROM_YEAR}, la 'probabilidad del ganador' debería ser > 0.5):")
report("Elo general", v["elo_prob_w"])
vs = v.dropna(subset=["elo_surf_prob_w"])
mix = (0.5 * vs["elo_prob_w"] + 0.5 * vs["elo_surf_prob_w"])
report("Elo de la superficie", vs["elo_surf_prob_w"])
report("Media Elo general + superficie", mix)
r = v.dropna(subset=["winner_rank", "loser_rank"])
r = r[r["winner_rank"] != r["loser_rank"]]
print(f"{'Mejor ranking ATP gana':34s} acierto: {(r['winner_rank'] < r['loser_rank']).mean():6.1%}")

print("\nTop 10 Elo entre jugadores activos en los últimos 12 meses del dataset:")
recent = final[final["last_match"] >= final["last_match"].max() - pd.Timedelta(days=365)]
print(recent.head(10)[["name", "elo", "elo_hard", "elo_clay", "elo_grass", "n_matches"]].round(0).to_string(index=False))