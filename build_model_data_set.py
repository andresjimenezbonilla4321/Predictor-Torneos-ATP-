"""
Convierte los partidos (ganador/perdedor) al formato que necesita el modelo.

Entrada : atp_matches_serve.csv   (salida de add_serve_stats.py)
Salida  : model_dataset.csv   (una fila por partido, con columna 'split')

Ejecutar:  python build_model_dataset.py

Qué hace:
  1. Reparte a los dos jugadores en "A" y "B" al azar (así el modelo no puede
     aprender que el primer jugador siempre gana).
  2. target = 1 si gana A, 0 si gana B.
  3. Las características se expresan como DIFERENCIAS (A - B).
  4. Marca cada partido como train / val / test según su fecha (nunca al azar).
"""
import numpy as np
import pandas as pd

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_serve.csv"
OUTPUT = "model_dataset.csv"

SEED = 42                 # fija el reparto aleatorio para poder repetirlo
FIRST_YEAR = 2005         # los años anteriores son el "calentamiento" del Elo
DROP_RETIREMENTS = True   # quitar partidos con retirada (resultado poco predecible)
TRAIN_END_YEAR = 2021     # train: FIRST_YEAR..2021
VAL_END_YEAR = 2023       # val: 2022-2023 | test: 2024 en adelante
UNRANKED_VALUE = 1500     # ranking que se asigna a quien no tiene ranking ATP
# -----------------------------------------------------------------

m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
m = m[m["tourney_date"].dt.year >= FIRST_YEAR]
if DROP_RETIREMENTS:
    m = m[m["is_retirement"] == 0]
m = m.reset_index(drop=True)
print(f"Partidos usados: {len(m):,}")

# ---------- 1) Reparto aleatorio de A y B ----------
rng = np.random.default_rng(SEED)
swap = rng.random(len(m)) < 0.5      # True -> A es el perdedor, B el ganador


def pick(winner_col, loser_col):
    """Devuelve (valor de A, valor de B) según el reparto aleatorio."""
    w, l = m[winner_col].to_numpy(), m[loser_col].to_numpy()
    return np.where(swap, l, w), np.where(swap, w, l)


A, B = {}, {}
for name, (wc, lc) in {
    "id": ("winner_id", "loser_id"),
    "name": ("winner_name", "loser_name"),
    "rank": ("winner_rank", "loser_rank"),
    "age": ("winner_age", "loser_age"),
    "ht": ("winner_ht", "loser_ht"),
    "hand": ("winner_hand", "loser_hand"),
    "elo": ("elo_w", "elo_l"),
    "elo_surf": ("elo_surf_w", "elo_surf_l"),
    "n": ("n_w", "n_l"),
    "form": ("form_w", "form_l"),
    "rest": ("rest_w", "rest_l"),
    "g14": ("games14_w", "games14_l"),
    "h2h": ("h2h_w", "h2h_l"),    # victorias de cada uno sobre el otro (ojo: cruzado, ver abajo)
    "spw": ("spw_w", "spw_l"),
    "rpw": ("rpw_w", "rpw_l"),
    "ace": ("ace_w", "ace_l"),
    "df": ("df_w", "df_l"),
    "fin": ("fin_w", "fin_l"),
    "ns": ("ns_w", "ns_l"),
}.items():
    A[name], B[name] = pick(wc, lc)

# ---------- 2) Target ----------
target = (~swap).astype(int)         # 1 si A es el ganador real

# ---------- 3) Características como diferencias (A - B) ----------
rankA = pd.Series(A["rank"], dtype=float).fillna(UNRANKED_VALUE)
rankB = pd.Series(B["rank"], dtype=float).fillna(UNRANKED_VALUE)

out = pd.DataFrame({
    "match_date": m["tourney_date"],
    "tourney_id": m["tourney_id"],
    "tourney_name": m["tourney_name"],
    "tourney_level": m["tourney_level"],
    "level_source": m["level_source"],
    "surface": m["surface"],
    "round": m["round"],
    "best_of": m["best_of"],
    "player_A_id": A["id"], "player_A_name": A["name"],
    "player_B_id": B["id"], "player_B_name": B["name"],
    # --- features (diferencias A - B) ---
    "elo_diff": A["elo"] - B["elo"],
    "elo_surf_diff": A["elo_surf"] - B["elo_surf"],
    # se usa el logaritmo: pasar del puesto 1 al 10 pesa más que del 101 al 110
    "log_rank_diff": np.log(rankA) - np.log(rankB),
    "age_diff": pd.Series(A["age"], dtype=float) - pd.Series(B["age"], dtype=float),
    "ht_diff": pd.Series(A["ht"], dtype=float) - pd.Series(B["ht"], dtype=float),
    "lefty_diff": (pd.Series(A["hand"]) == "L").astype(int) - (pd.Series(B["hand"]) == "L").astype(int),
    # experiencia: cuánto se puede fiar del Elo de cada uno
    "exp_diff": np.log1p(pd.Series(A["n"], dtype=float)) - np.log1p(pd.Series(B["n"], dtype=float)),
    # --- historial ---
    "form_diff": pd.Series(A["form"], dtype=float) - pd.Series(B["form"], dtype=float),
    "rest_diff": pd.Series(A["rest"], dtype=float) - pd.Series(B["rest"], dtype=float),
    "games14_diff": pd.Series(A["g14"], dtype=float) - pd.Series(B["g14"], dtype=float),
    # cara a cara suavizado: (victorias de A - victorias de B) / (total + 2)
    "h2h_diff": (pd.Series(A["h2h"], dtype=float) - pd.Series(B["h2h"], dtype=float))
                / (pd.Series(A["h2h"], dtype=float) + pd.Series(B["h2h"], dtype=float) + 2),
    # --- saque y resto (historial de partidos anteriores) ---
    "spw_diff": pd.Series(A["spw"], dtype=float) - pd.Series(B["spw"], dtype=float),
    "rpw_diff": pd.Series(A["rpw"], dtype=float) - pd.Series(B["rpw"], dtype=float),
    "ace_diff": pd.Series(A["ace"], dtype=float) - pd.Series(B["ace"], dtype=float),
    "df_diff": pd.Series(A["df"], dtype=float) - pd.Series(B["df"], dtype=float),
    "fin_diff": pd.Series(A["fin"], dtype=float) - pd.Series(B["fin"], dtype=float),
    # probabilidad que daba la fórmula del Elo (para comparar después)
    "elo_prob_A": 1 / (1 + 10 ** (-(A["elo"] - B["elo"]) / 400)),
    "target": target,
})

# alturas/edades que faltan -> diferencia 0 ("sin información")
out[["age_diff", "ht_diff"]] = out[["age_diff", "ht_diff"]].fillna(0)

# ---------- 4) División temporal ----------
year = out["match_date"].dt.year
out["split"] = np.select([year <= TRAIN_END_YEAR, year <= VAL_END_YEAR], ["train", "val"], default="test")

out.to_csv(OUTPUT, index=False)

print("\nPartidos por conjunto:")
print(out["split"].value_counts().reindex(["train", "val", "test"]).to_string())
print("\nRango de fechas:")
print(out.groupby("split")["match_date"].agg(["min", "max"]).reindex(["train", "val", "test"]).to_string())
print(f"\nTarget medio (debería rondar 0.50): {out['target'].mean():.3f}")
print(f"Valores vacíos en features: {int(out.filter(like='_diff').isna().sum().sum())}")
print(f"Archivo guardado: {OUTPUT}")