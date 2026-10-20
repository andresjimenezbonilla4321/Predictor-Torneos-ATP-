"""
Limpieza de atp_matches.csv  ->  atp_matches_clean.csv

Ejecutar en la carpeta donde esté atp_matches.csv:
    python clean_matches.py

Cada regla se registra en el informe final para que sea reproducible.
"""
import re
import numpy as np
import pandas as pd

# ------------------------- CONFIGURACIÓN -------------------------
INPUT = "atp_matches.csv"
OUTPUT = "atp_matches_clean.csv"

DROP_QUALIFYING = True    # True = quitar rondas Q1/Q2/Q3 (fase previa)
DROP_DAVIS_CUP = True     # True = quitar la Copa Davis (nivel D)
CHALLENGER_MIN_YEAR = 2000  # p. ej. 2010 = quitar Challengers anteriores a ese año
# -----------------------------------------------------------------

report = []


def log(msg):
    report.append(msg)
    print(msg)


m = pd.read_csv(INPUT, low_memory=False)
n0 = len(m)
log(f"Filas de partida: {n0:,}")

# ---------- 1. Fechas ----------
m["tourney_date"] = pd.to_datetime(m["tourney_date"].astype(str), format="%Y%m%d")

# ---------- 2. Errores estructurales ----------
mask = m["winner_id"] == m["loser_id"]
log(f"[1] Partidos con el mismo jugador como ganador y perdedor (eliminados): {mask.sum()}")
m = m[~mask]

key = ["tourney_id", "winner_id", "loser_id", "round", "score"]
mask = m.duplicated(key, keep="first")
log(f"[2] Partidos duplicados (mismo torneo, jugadores, ronda y resultado) eliminados: {mask.sum()}")
m = m[~mask]

# ---------- 3. Resultado del partido ----------
score = m["score"].fillna("").astype(str)
is_wo = score.str.contains(r"W/O|Walkover", case=False, regex=True) | (score.str.strip() == "")
is_unfinished = score.str.contains(r"unfinished|abandoned", case=False, regex=True)
is_ret = score.str.contains(r"RET|DEF|Default", case=False, regex=True) & ~is_wo & ~is_unfinished

log(f"[3] Walkovers / sin resultado (eliminados): {is_wo.sum()}")
log(f"[4] 'Played and unfinished/abandoned' (eliminados): {is_unfinished.sum()}")
log(f"[5] Retiradas (RET/DEF): {is_ret.sum()} -> se mantienen con is_retirement=1")
m = m.assign(is_retirement=is_ret.astype(int))[~(is_wo | is_unfinished)].copy()

# ---------- 4. Marcas útiles ----------
m["is_qualifying"] = m["round"].isin(["Q1", "Q2", "Q3"]).astype(int)
m["is_davis_cup"] = (m["tourney_level"] == "D").astype(int)

if DROP_QUALIFYING:
    log(f"[6] Rondas de clasificación eliminadas: {m['is_qualifying'].sum()}")
    m = m[m["is_qualifying"] == 0]
if DROP_DAVIS_CUP:
    log(f"[7] Copa Davis eliminada: {m['is_davis_cup'].sum()}")
    m = m[m["is_davis_cup"] == 0]

if CHALLENGER_MIN_YEAR > 2000:
    mask = (m["tourney_level"] == "C") & (m["tourney_date"].dt.year < CHALLENGER_MIN_YEAR)
    log(f"[7b] Challengers anteriores a {CHALLENGER_MIN_YEAR} eliminados: {mask.sum()}")
    m = m[~mask]

# ---------- 5. Valores imposibles -> NaN ----------
for side in ["winner", "loser"]:
    # mano desconocida
    m[f"{side}_hand"] = m[f"{side}_hand"].replace({"U": np.nan})
    # altura imposible (adultos 160-215 cm)
    bad = ~m[f"{side}_ht"].between(160, 215) & m[f"{side}_ht"].notna()
    log(f"[8] {side}_ht fuera de 160-215 cm -> NaN: {bad.sum()}")
    m.loc[bad, f"{side}_ht"] = np.nan
    # edad imposible
    bad = ~m[f"{side}_age"].between(14, 50) & m[f"{side}_age"].notna()
    log(f"[9] {side}_age fuera de 14-50 -> NaN: {bad.sum()}")
    m.loc[bad, f"{side}_age"] = np.nan

# entradas al cuadro: valores corruptos y variantes
entry_map = {"Alt": "ALT", "AL": "ALT"}
valid_entry = {"Q", "WC", "ALT", "LL", "PR", "SE", "ITF", "NG", "JR", "CO", "S", "P", "A", "W", "UP", "I"}
for side in ["winner", "loser"]:
    col = f"{side}_entry"
    m[col] = m[col].replace(entry_map)
    bad = m[col].notna() & ~m[col].isin(valid_entry)
    log(f"[10] {col} corrupto -> NaN: {bad.sum()}")
    m.loc[bad, col] = np.nan

# seeds mayores que el cuadro
for side in ["winner", "loser"]:
    bad = m[f"{side}_seed"] > m["draw_size"]
    m.loc[bad, f"{side}_seed"] = np.nan

# ---------- 6. Minutos ----------
def total_games(sc):
    sc = re.sub(r"\[.*?\]", "", str(sc))
    return sum(int(a) + int(b) for a, b in re.findall(r"(\d+)-(\d+)", sc))


games = m["score"].apply(total_games)
mpg = m["minutes"] / games.replace(0, np.nan)
bad = m["minutes"].notna() & ((m["minutes"] <= 0) | (mpg < 1.5) | (mpg > 9))
log(f"[11] Minutos incoherentes con el marcador (<1,5 o >9 min/juego) -> NaN: {bad.sum()}")
m.loc[bad, "minutes"] = np.nan
# en retiradas el tiempo no es comparable con partidos completos
m.loc[m["is_retirement"] == 1, "minutes"] = np.nan

# ---------- 7. Estadísticas de saque ----------
stat_cols = [c for c in m.columns if c.startswith(("w_", "l_"))]
bad = (m["w_svpt"] == 0) | (m["l_svpt"] == 0)
bad |= (m["w_2ndWon"] > (m["w_svpt"] - m["w_1stIn"])) | (m["l_2ndWon"] > (m["l_svpt"] - m["l_1stIn"]))
log(f"[12] Partidos con estadísticas incoherentes (0 puntos de saque, etc.) -> stats a NaN: {bad.sum()}")
m.loc[bad, stat_cols] = np.nan
m["has_stats"] = m["w_svpt"].notna().astype(int)

# ---------- 8. Orden cronológico ----------
round_order = {"Q1": 0, "Q2": 1, "Q3": 2, "RR": 3, "ER": 3, "R128": 4, "R64": 5, "R32": 6,
               "R16": 7, "QF": 8, "SF": 9, "BR": 10, "F": 11}
m["round_order"] = m["round"].map(round_order)
m = m.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"]).reset_index(drop=True)

# ---------- 9. Columnas ----------
m = m.drop(columns=["source_year"])

m.to_csv(OUTPUT, index=False)
log(f"\nFilas finales: {len(m):,} (eliminadas {n0 - len(m):,})")
log(f"Rango de fechas: {m['tourney_date'].min().date()} -> {m['tourney_date'].max().date()}")
log(f"Archivo guardado: {OUTPUT}")

with open("cleaning_report_matches.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))