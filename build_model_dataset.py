"""
Convierte los partidos (ganador/perdedor) al formato que necesita el modelo.

Entrada : atp_matches_serve.csv   (salida de add_serve_stats.py)
Salida  : model_dataset.csv   (una fila por partido, con columna 'split')

Ejecutar:  python build_model_dataset.py

Qué hace:
  1. Reparte a los dos jugadores en "A" y "B" al azar (así el modelo no puede
     aprender que el primer jugador siempre gana).
  2. target = 1 si gana A, 0 si gana B.
  3. Las características se expresan como DIFERENCIAS (A - B), con common.make_features
     (la misma función que usa predict.py para partidos futuros).
  4. Marca cada partido como train / val / test según su fecha (nunca al azar).
  5. Las retiradas se quitan solo de train. En val/test se mantienen (con is_retirement=1),
     porque antes de un partido no se sabe si habrá retirada: así la evaluación es realista.
"""
import numpy as np
import pandas as pd

from common import FIRST_YEAR, TRAIN_END_YEAR, VAL_END_YEAR, make_features

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches_serve.csv"
OUTPUT = "model_dataset.csv"

SEED = 42                       # fija el reparto aleatorio para poder repetirlo
DROP_RETIREMENTS_TRAIN = True   # quitar retiradas del entrenamiento (resultado poco informativo)
# -----------------------------------------------------------------

# valor de cada jugador: (columna del ganador, columna del perdedor)
PLAYER_COLS = {
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
    "layoff": ("layoff_w", "layoff_l"),
    "g14": ("games14_w", "games14_l"),
    "h2h": ("h2h_w", "h2h_l"),       # victorias de cada uno sobre el otro
    "spw": ("spw_w", "spw_l"),
    "rpw": ("rpw_w", "rpw_l"),
    "ace": ("ace_w", "ace_l"),
    "df": ("df_w", "df_l"),
    "fin": ("fin_w", "fin_l"),
}


def main():
    m = pd.read_csv(INPUT, low_memory=False, parse_dates=["tourney_date"])
    m = m[m["tourney_date"].dt.year >= FIRST_YEAR]
    if DROP_RETIREMENTS_TRAIN:
        m = m[~((m["is_retirement"] == 1) & (m["tourney_date"].dt.year <= TRAIN_END_YEAR))]
    m = m.reset_index(drop=True)
    print(f"Partidos usados: {len(m):,}")

    # ---------- 1) Reparto aleatorio de A y B ----------
    rng = np.random.default_rng(SEED)
    swap = rng.random(len(m)) < 0.5      # True -> A es el perdedor, B el ganador

    A, B = {}, {}
    for name, (wc, lc) in PLAYER_COLS.items():
        w, l = m[wc].to_numpy(), m[lc].to_numpy()
        A[name], B[name] = np.where(swap, l, w), np.where(swap, w, l)

    # ---------- 2) Target ----------
    target = (~swap).astype(int)         # 1 si A es el ganador real

    # ---------- 3) Características como diferencias (A - B) ----------
    feats = make_features(A, B, m["best_of"].to_numpy())

    out = pd.DataFrame({
        "match_date": m["tourney_date"],
        "tourney_id": m["tourney_id"],
        "tourney_name": m["tourney_name"],
        "tourney_level": m["tourney_level"],
        "level_source": m["level_source"],
        "surface": m["surface"],
        "round": m["round"],
        "best_of": m["best_of"],
        "is_retirement": m["is_retirement"],
        "player_A_id": A["id"], "player_A_name": A["name"],
        "player_B_id": B["id"], "player_B_name": B["name"],
        **feats,
        # probabilidad que daba la fórmula del Elo (para comparar después)
        "elo_prob_A": 1 / (1 + 10 ** (-(A["elo"].astype(float) - B["elo"].astype(float)) / 400)),
        "target": target,
    })

    # ---------- 4) División temporal ----------
    year = out["match_date"].dt.year
    out["split"] = np.select([year <= TRAIN_END_YEAR, year <= VAL_END_YEAR], ["train", "val"], default="test")

    out.to_csv(OUTPUT, index=False)

    print("\nPartidos por conjunto:")
    print(out["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("\nRango de fechas:")
    print(out.groupby("split")["match_date"].agg(["min", "max"]).reindex(["train", "val", "test"]).to_string())
    print(f"\nTarget medio (debería rondar 0.50): {out['target'].mean():.3f}")
    print(f"Valores vacíos en features: {int(out[list(feats)].isna().sum().sum())}")
    print(f"Archivo guardado: {OUTPUT}")


if __name__ == "__main__":
    main()
