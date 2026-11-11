"""
Predice partidos FUTUROS con el modelo final y el estado actual de cada jugador.

Necesita haber ejecutado el pipeline (python run_pipeline.py), que deja:
  state/elo.pkl, state/history.pkl, state/serve.pkl, models/best_model.joblib
  atp_players.csv, atp_rankings.csv

Uso:
  python predict.py "Jannik Sinner" "Carlos Alcaraz" --surface Clay --best-of 5
  python predict.py Sinner Alcaraz --surface Grass --date 2026-06-29

Desde Python:
  from predict import Predictor
  pr = Predictor(date="2026-06-08")
  pr.match_proba("Sinner", "Alcaraz", surface="Clay", best_of=5)

Las características se calculan con las MISMAS funciones de common.py que se usaron
para entrenar, a partir del estado de cada jugador al final de los datos.
"""
import argparse
import unicodedata

import joblib
import numpy as np
import pandas as pd

from common import (LAYOFF_CAP, MODELS_DIR, START_RATING, expected, history_state, inactivity_shrink,
                    load_state, make_features, serve_state, to_day)

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
DEFAULT_DAYS_AFTER_DATA = 7    # sin --date: la semana siguiente al último torneo de los datos


def normalize(text):
    """Minúsculas y sin tildes: 'Núñez' -> 'nunez'."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


class Predictor:
    def __init__(self, date=None, model_path=MODELS_DIR / "best_model.joblib"):
        self.elo = load_state("elo")
        self.hist = load_state("history")
        self.serve = load_state("serve")
        bundle = joblib.load(model_path)
        self.model, self.features, self.model_name = bundle["model"], bundle["features"], bundle["name"]

        self.data_end = pd.Timestamp(self.elo["last_date"])
        self.date = pd.Timestamp(date) if date else self.data_end + pd.Timedelta(days=DEFAULT_DAYS_AFTER_DATA)
        if self.date < self.data_end:
            print(f"Aviso: la fecha {self.date.date()} es anterior al fin de los datos ({self.data_end.date()}); "
                  "el estado de los jugadores ya incluye partidos posteriores.")
        self.day = int(to_day([self.date.to_datetime64()])[0])

        # jugadores
        p = pd.read_csv("atp_players.csv", low_memory=False)
        p["name"] = (p["name_first"].fillna("") + " " + p["name_last"].fillna("")).str.strip()
        p["key"] = p["name"].map(normalize)
        p["n_matches"] = p["player_id"].map(self.elo["gen_n"]).fillna(0).astype(int)
        self.players = p.set_index("player_id")

        # ranking: la última semana publicada antes de la fecha
        r = pd.read_csv("atp_rankings.csv")
        date_int = int(self.date.strftime("%Y%m%d"))
        last = r.loc[r["ranking_date"] <= date_int, "ranking_date"].max()
        self.ranking_date = last
        self.rank = r[r["ranking_date"] == last].drop_duplicates("player").set_index("player")["rank"].to_dict()

    # ------------------------- buscar jugadores -------------------------
    def find_player(self, query):
        """Devuelve el player_id que mejor encaja con el nombre (o el id si se pasa un número)."""
        if isinstance(query, (int, np.integer)) or str(query).isdigit():
            return int(query)
        q = normalize(query)
        p = self.players
        cand = p[p["key"] == q]
        if cand.empty:
            words = q.split()
            mask = np.ones(len(p), dtype=bool)
            for w in words:
                mask &= p["key"].str.contains(w, regex=False).to_numpy()
            cand = p[mask]
        if cand.empty:
            raise ValueError(f"No encuentro al jugador '{query}'")
        # si hay varios, el que más partidos tiene en los datos
        cand = cand.sort_values("n_matches", ascending=False)
        # solo se avisa si el segundo candidato también tiene bastantes partidos
        if len(cand) > 1 and cand["n_matches"].iloc[1] >= max(1, 0.2 * cand["n_matches"].iloc[0]):
            others = ", ".join(cand["name"].iloc[1:4])
            print(f"Aviso: '{query}' es ambiguo; uso {cand['name'].iloc[0]} (otros: {others})")
        return int(cand.index[0])

    def name(self, pid):
        return self.players.at[pid, "name"] if pid in self.players.index else str(pid)

    # ------------------------- estado de un jugador -------------------------
    def profile(self, pid, surface):
        """Valores del jugador el día de la predicción (lo mismo que ve el modelo al entrenar)."""
        d = self.day
        e = self.elo
        prm = e["params"]
        last = e["last_day"].get(pid)
        gap = d - last if last is not None else 0

        elo = inactivity_shrink(e["gen_r"].get(pid, START_RATING), gap, prm["min_days"], prm["shrink"])
        if surface in SURFACES and (pid, surface) in e["srf_r"]:
            elo_surf = inactivity_shrink(e["srf_r"][(pid, surface)], gap, prm["min_days"], prm["shrink"])
        elif surface in SURFACES:
            elo_surf = elo if prm["surf_init"] else START_RATING
        else:
            elo_surf = np.nan

        recent = [(dd, g) for (dd, g) in self.hist["recent"].get(pid, []) if d - dd <= LAYOFF_CAP + 1]
        form, rest, layoff, g14 = history_state(recent, self.hist["results"].get(pid, []), d)
        if not recent and pid in self.hist["last_day"]:
            layoff = min(d - self.hist["last_day"][pid], LAYOFF_CAP)

        spw, rpw, ace, df, fin, ns = serve_state(self.serve["hist"].get(pid, []), self.serve["prior"])

        info = self.players.loc[pid] if pid in self.players.index else None
        dob = pd.to_datetime(str(int(info["dob"])), format="%Y%m%d", errors="coerce") \
            if info is not None and pd.notna(info["dob"]) else pd.NaT
        age = (self.date - dob).days / 365.25 if pd.notna(dob) else np.nan
        ht = float(info["height"]) if info is not None and pd.notna(info["height"]) else np.nan
        hand = info["hand"] if info is not None and info["hand"] in ("L", "R") else np.nan

        return {"id": pid, "elo": elo, "elo_surf": elo_surf, "rank": self.rank.get(pid, np.nan),
                "age": age, "ht": ht if 160 <= ht <= 215 else np.nan, "hand": hand,
                "n": e["gen_n"].get(pid, 0), "form": form, "rest": rest, "layoff": layoff, "g14": g14,
                "spw": spw, "rpw": rpw, "ace": ace, "df": df, "fin": fin, "n_serve": ns}

    # ------------------------- probabilidades -------------------------
    def pairs_proba(self, pairs, surface, best_of, profiles=None):
        """Probabilidad de que gane el primero en cada pareja (a_id, b_id)."""
        profiles = profiles or {}
        for pid in {x for pair in pairs for x in pair}:
            if pid not in profiles:
                profiles[pid] = self.profile(pid, surface)
        keys = ["elo", "elo_surf", "rank", "age", "ht", "hand", "n", "form", "rest", "layoff", "g14",
                "spw", "rpw", "ace", "df", "fin"]
        A = {k: np.array([profiles[a][k] for a, _ in pairs], dtype=object if k == "hand" else float) for k in keys}
        B = {k: np.array([profiles[b][k] for _, b in pairs], dtype=object if k == "hand" else float) for k in keys}
        h2h = self.hist["h2h"]
        A["h2h"] = np.array([h2h.get((a, b), 0) for a, b in pairs], dtype=float)
        B["h2h"] = np.array([h2h.get((b, a), 0) for a, b in pairs], dtype=float)
        X = pd.DataFrame(make_features(A, B, np.full(len(pairs), best_of)))
        return self.model.predict(X[self.features])

    def match_proba(self, player_a, player_b, surface="Hard", best_of=3):
        a, b = self.find_player(player_a), self.find_player(player_b)
        return float(self.pairs_proba([(a, b)], surface, best_of)[0])


def main():
    ap = argparse.ArgumentParser(description="Probabilidad de victoria en un partido ATP futuro")
    ap.add_argument("player_a")
    ap.add_argument("player_b")
    ap.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--date", help="fecha del partido (AAAA-MM-DD). Por defecto: 1 semana tras el fin de los datos")
    args = ap.parse_args()

    pr = Predictor(date=args.date)
    a, b = pr.find_player(args.player_a), pr.find_player(args.player_b)
    p = float(pr.pairs_proba([(a, b)], args.surface, args.best_of)[0])
    pa, pb = pr.profile(a, args.surface), pr.profile(b, args.surface)
    na, nb = pr.name(a), pr.name(b)

    print(f"\n{na} vs {nb}  |  {args.surface}, al mejor de {args.best_of}  |  fecha {pr.date.date()}")
    print(f"Modelo: {pr.model_name}  |  datos hasta {pr.data_end.date()}  |  ranking del {pr.ranking_date}\n")
    rows = [("Elo general", "elo", "{:.0f}"), (f"Elo {args.surface}", "elo_surf", "{:.0f}"),
            ("Ranking ATP", "rank", "{:.0f}"), ("Forma (últ. 10, suavizada)", "form", "{:.2f}"),
            ("Días sin jugar", "layoff", "{:.0f}"), ("% puntos ganados al saque", "spw", "{:.3f}"),
            ("% puntos ganados al resto", "rpw", "{:.3f}"), ("Partidos en los datos", "n", "{:.0f}")]
    print(f"  {'':28s} {na[:22]:>22s} {nb[:22]:>22s}")
    for label, k, fmt in rows:
        fa = fmt.format(pa[k]) if pd.notna(pa[k]) else "-"
        fb = fmt.format(pb[k]) if pd.notna(pb[k]) else "-"
        print(f"  {label:28s} {fa:>22s} {fb:>22s}")
    print(f"  {'Cara a cara (victorias)':28s} {pr.hist['h2h'].get((a, b), 0):>22d} {pr.hist['h2h'].get((b, a), 0):>22d}")
    print(f"\n  Solo Elo:  {na} {expected(pa['elo'], pb['elo']) * 100:.1f} %")
    print(f"  MODELO:    {na} {p * 100:.1f} %   |   {nb} {(1 - p) * 100:.1f} %")


if __name__ == "__main__":
    main()
