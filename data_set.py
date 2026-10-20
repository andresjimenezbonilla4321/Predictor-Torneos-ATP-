import pandas as pd

BASE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/"

START_YEAR = 2000
END_YEAR = 2026
INCLUDE_FUTURES = False

YEARS = range(START_YEAR, END_YEAR + 1)
failed = []


def load_years(prefix):
    dfs = []
    for y in YEARS:
        url = f"{BASE}{prefix}_{y}.csv"
        try:
            df = pd.read_csv(url, low_memory=False)
            df["source_year"] = y
            dfs.append(df)
        except Exception as e:
            failed.append((prefix, y, type(e).__name__))
    if not dfs:
        raise RuntimeError(f"No se pudo descargar nada para '{prefix}'. Ejemplo de URL: {url}")
    return pd.concat(dfs, ignore_index=True)


# Partidos
main = load_years("atp_matches")
main["level_source"] = "main"

chall = load_years("atp_matches_qual_chall")
chall["level_source"] = "challenger_qual"

frames = [main, chall]

if INCLUDE_FUTURES:
    futures = load_years("atp_matches_futures")
    futures["level_source"] = "futures"
    frames.append(futures)

matches = pd.concat(frames, ignore_index=True)

# Jugadores
players = pd.read_csv(BASE + "atp_players.csv", low_memory=False)

# Rankings: solo desde START_YEAR
rank_files = ["00s", "10s", "20s", "current"]
rankings = pd.concat(
    [pd.read_csv(f"{BASE}atp_rankings_{r}.csv", low_memory=False) for r in rank_files],
    ignore_index=True,
)
rankings = rankings[rankings["ranking_date"] >= START_YEAR * 10000]
rankings = rankings.drop_duplicates().reset_index(drop=True)

# Guardar en CSV
matches.to_csv("atp_matches.csv", index=False)
players.to_csv("atp_players.csv", index=False)
rankings.to_csv("atp_rankings.csv", index=False)

print("matches:", matches.shape, "| players:", players.shape, "| rankings:", rankings.shape)
print(matches["level_source"].value_counts())
print("Descargas fallidas:", failed if failed else "ninguna")