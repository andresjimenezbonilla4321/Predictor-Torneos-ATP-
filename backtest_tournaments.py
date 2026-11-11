"""
Backtest del simulador: simula los torneos REALES de test (2024 en adelante) con lo que
se sabía ANTES de empezar cada uno y compara las predicciones con lo que pasó.

  ¿Los jugadores a los que dábamos un 20 % de ser campeones ganaron uno de cada cinco?
  ¿Es mejor el modelo que simular solo con el Elo?

Cómo se evita la fuga de datos:
  - Modelo: models/model_trainval.joblib, entrenado y calibrado solo con 2005-2023.
  - Estado de cada jugador: las columnas "antes del partido" de su PRIMER partido en el
    torneo (Elo, forma, saque, ranking...). Es exactamente lo que se sabía al empezar.
  - Cara a cara: se cuenta solo con los partidos anteriores al torneo.
  - El cuadro se reconstruye de los resultados, pero solo se usa su ESTRUCTURA (quién
    está en cada posición); los resultados únicamente se usan para evaluar.
  - Los walkovers se simulan como partidos normales: la predicción se hace al publicarse el cuadro.

Entradas : atp_matches.csv, atp_matches_serve.csv, state/serve.pkl, models/model_trainval.joblib
Salidas  : backtest_report.txt, backtest_tournaments.csv

Ejecutar:
  python backtest_tournaments.py                                 # todos los torneos de test
  python backtest_tournaments.py --show "Roland Garros" --year 2025   # detalle de un torneo
"""
import argparse
import math

import joblib
import numpy as np
import pandas as pd

from build_model_dataset import PLAYER_COLS
from common import (LAYOFF_CAP, MODELS_DIR, REST_CAP, START_RATING, TRAIN_END_YEAR, VAL_END_YEAR,
                    CalibratedModel, expected, load_state, make_features)
from simulate_tournament import BYE, round_names, simulate

# ------------------------- CONFIGURACIÓN -------------------------
SIMS = 10000
SEED = 0
ROUND_SEQ = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
LEVELS = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 250/500", "C": "Challenger"}
PROFILE_KEYS = ["elo", "elo_surf", "rank", "age", "ht", "hand", "n", "form", "rest", "layoff", "g14",
                "spw", "rpw", "ace", "df", "fin"]
# -----------------------------------------------------------------


class BracketError(ValueError):
    pass


def build_bracket(rows):
    """
    Reconstruye el cuadro a partir de los partidos de un torneo.
    rows: lista de (ronda, ganador, perdedor).
    Devuelve (slots, rondas, alcanzado):
      slots     : lista de 2^K posiciones (id de jugador o BYE) en orden de cuadro
      rondas    : rondas del torneo, p. ej. ["R32", "R16", "QF", "SF", "F"]
      alcanzado : {jugador: ronda alcanzada (0 = primera ronda ... K = campeón)}
    """
    present = sorted({r for r, _, _ in rows}, key=lambda r: ROUND_SEQ.index(r) if r in ROUND_SEQ else -1)
    if any(r not in ROUND_SEQ for r in present):
        raise BracketError(f"rondas no eliminatorias: {present}")
    idx = [ROUND_SEQ.index(r) for r in present]
    if present[-1] != "F" or idx != list(range(idx[0], idx[0] + len(idx))):
        raise BracketError(f"rondas incompletas: {present}")
    K = len(present)

    won = {}
    for r, w, l in rows:
        k = present.index(r)
        if (k, w) in won:
            raise BracketError(f"el jugador {w} gana dos partidos en {r}")
        won[(k, w)] = (w, l)
    finals = [(w, l) for r, w, l in rows if r == "F"]
    if len(finals) != 1:
        raise BracketError(f"{len(finals)} finales")

    used = [0]

    def expand(p, k):
        """Posiciones del subcuadro (tamaño 2^k) del que sale el jugador p al llegar a la ronda k."""
        if k == 0:
            return [p]
        m = won.get((k - 1, p))
        if m is None:      # p no jugó la ronda anterior: tuvo un BYE (o entró directamente más tarde)
            return expand(p, k - 1) + [BYE] * (2 ** (k - 1))
        used[0] += 1
        return expand(m[0], k - 1) + expand(m[1], k - 1)

    w, l = finals[0]
    used[0] += 1
    slots = expand(w, K - 1) + expand(l, K - 1)

    players = [s for s in slots if s != BYE]
    if used[0] != len(rows) or len(set(players)) != len(players):
        raise BracketError("estructura inconsistente (partidos sin colocar o jugadores repetidos)")

    reached = {}
    for r, w_, l_ in rows:
        k = present.index(r)
        reached[l_] = max(reached.get(l_, 0), k)
        reached[w_] = max(reached.get(w_, 0), k + 1)
    return slots, present, reached


def long_profiles(m):
    """Una fila por (partido, jugador) con su estado ANTES del partido."""
    parts = []
    for side, pick in (("w", 0), ("l", 1)):
        cols = {k: v[pick] for k, v in PLAYER_COLS.items()}
        part = m[list(cols.values()) + ["tourney_id", "tourney_date", "round_order"]].copy()
        part.columns = list(cols.keys()) + ["tourney_id", "tourney_date", "round_order"]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    return long.sort_values(["tourney_date", "round_order"], kind="stable")


def load_model(split):
    """
    test -> modelo entrenado con 2005-2023 (calibrado en validación)
    val  -> modelo entrenado solo con 2005-2021 y sin calibrar
    Devuelve (modelo base sin calibrar, a de calibración por partido, features, nombre).
    """
    path = MODELS_DIR / ("model_trainval.joblib" if split == "test" else "model_train.joblib")
    bundle = joblib.load(path)
    model = bundle["model"]
    if isinstance(model, CalibratedModel):
        return model.base, model.a, bundle["features"], bundle["name"]
    return model, 1.0, bundle["features"], bundle["name"]


def prepare(split, show=None, year=None):
    """Torneos del periodo, estado previo de cada jugador y cara a cara antes de cada torneo."""
    raw = pd.read_csv("atp_matches.csv", low_memory=False)
    raw["tourney_date"] = pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d")
    yr = raw["tourney_date"].dt.year
    period = (yr > VAL_END_YEAR) if split == "test" else ((yr > TRAIN_END_YEAR) & (yr <= VAL_END_YEAR))
    raw = raw[period & ~raw["round"].isin(["Q1", "Q2", "Q3", "BR"]) & (raw["tourney_level"] != "D")]
    if show:
        raw = raw[raw["tourney_name"].str.contains(show, case=False, regex=False)]
        if year:
            raw = raw[raw["tourney_date"].dt.year == year]
        if raw.empty:
            raise SystemExit(f"No hay ningún torneo de {split} que contenga '{show}'.")

    m = pd.read_csv("atp_matches_serve.csv", low_memory=False, parse_dates=["tourney_date"])
    long = long_profiles(m)
    first_in_tourney = long.drop_duplicates(["tourney_id", "id"], keep="first").set_index(["tourney_id", "id"])
    by_player = {p: g for p, g in long.groupby("id")}

    targets = set(raw["tourney_id"])
    players_of = {t: set(g["winner_id"]) | set(g["loser_id"]) for t, g in raw.groupby("tourney_id")}
    h2h, snap = {}, {}
    tids, W, L = m["tourney_id"].to_numpy(), m["winner_id"].to_numpy(), m["loser_id"].to_numpy()
    for i in range(len(m)):
        t = tids[i]
        if t in targets and t not in snap:
            ps = players_of[t]
            snap[t] = {(a, b): h2h[(a, b)] for a in ps for b in ps if (a, b) in h2h}
        h2h[(W[i], L[i])] = h2h.get((W[i], L[i]), 0) + 1
    return raw, first_in_tourney, by_player, snap


def run_backtest(split, a_values, sims=SIMS, show=None, year=None):
    """
    Simula todos los torneos del periodo con cada valor de calibración a (p' = σ(a·logit p), aplicado
    a las probabilidades SIN calibrar del modelo). Devuelve ({a: (resultados, celdas)}, descartados, info).
    """
    base, a_match, features, model_name = load_model(split)
    prior = load_state("serve")["prior"]
    raw, first_in_tourney, by_player, snap = prepare(split, show, year)

    out = {a: ([], []) for a in a_values}
    skipped = []
    for t_index, (tid, g) in enumerate(raw.groupby("tourney_id", sort=False)):
        info = g.iloc[0]
        try:
            slots_ids, rounds, reached = build_bracket(list(zip(g["round"], g["winner_id"], g["loser_id"])))
        except BracketError as e:
            skipped.append((info["tourney_name"], str(info["tourney_date"].date()), str(e)))
            continue

        players = [p for p in slots_ids if p != BYE]
        date = info["tourney_date"]
        surface, best_of = info["surface"], int(info["best_of"])

        # ---------- estado de cada jugador al empezar el torneo ----------
        profiles, n_fallback = {}, 0
        for p in players:
            if (tid, p) in first_in_tourney.index:
                row = first_in_tourney.loc[(tid, p)]
            else:
                # solo jugó walkovers en este torneo: su último estado conocido antes del torneo
                n_fallback += 1
                hist = by_player.get(p)
                hist = hist[hist["tourney_date"] < date] if hist is not None else None
                row = hist.iloc[-1] if hist is not None and len(hist) else None
            if row is not None:
                profiles[p] = {k: row[k] for k in PROFILE_KEYS}
            else:   # nunca visto en los datos: jugador nuevo
                side = g[g["winner_id"] == p]
                pre = "winner" if len(side) else "loser"
                r0 = side.iloc[0] if len(side) else g[g["loser_id"] == p].iloc[0]
                profiles[p] = {"elo": START_RATING, "elo_surf": START_RATING, "rank": r0[f"{pre}_rank"],
                               "age": r0[f"{pre}_age"], "ht": r0[f"{pre}_ht"], "hand": r0[f"{pre}_hand"],
                               "n": 0, "form": 0.5, "rest": REST_CAP, "layoff": LAYOFF_CAP, "g14": 0,
                               **{k: prior[k] for k in ("spw", "rpw", "ace", "df", "fin")}}

        # ---------- probabilidades de todos los cruces ----------
        pos = {p: i for i, p in enumerate(players)}
        pairs = [(a, b) for a in players for b in players if a != b]
        A = {k: np.array([profiles[a][k] for a, _ in pairs], dtype=object if k == "hand" else float)
             for k in PROFILE_KEYS}
        B = {k: np.array([profiles[b][k] for _, b in pairs], dtype=object if k == "hand" else float)
             for k in PROFILE_KEYS}
        hs = snap.get(tid, {})
        A["h2h"] = np.array([hs.get((a, b), 0) for a, b in pairs], dtype=float)
        B["h2h"] = np.array([hs.get((b, a), 0) for a, b in pairs], dtype=float)
        X = pd.DataFrame(make_features(A, B, np.full(len(pairs), best_of)))[features]
        p_raw = base.predict(X)
        p_elo = expected(A["elo"], B["elo"])

        n = len(players)
        slots = np.array([BYE if p == BYE else pos[p] for p in slots_ids])
        ia = np.array([pos[a] for a, _ in pairs])
        ib = np.array([pos[b] for _, b in pairs])
        # misma semilla por torneo para todos los valores de a: así las diferencias entre
        # calibraciones no son ruido de la simulación
        seed = SEED + t_index

        def reach_for(pp):
            prob = np.full((n, n), 0.5)
            prob[ia, ib] = pp
            return simulate(prob, slots, sims, np.random.default_rng(seed)) / sims

        reach_elo = reach_for(p_elo)
        K = len(rounds)
        actual = np.zeros((n, K + 1))
        for p in players:
            actual[pos[p], :reached.get(p, 0) + 1] = 1
        champ = next(p for p in players if reached.get(p) == K)
        level = LEVELS.get(info["tourney_level"], "Otros")

        for a in a_values:
            reach_m = reach_for(CalibratedModel.apply(p_raw, a))
            fav = players[int(np.argmax(reach_m[:, K]))]
            out[a][0].append({
                "tourney_id": tid, "tourney_name": info["tourney_name"], "date": date.date(), "level": level,
                "surface": surface, "players": n, "draw": len(slots_ids), "fallback_profiles": n_fallback,
                "champion": champ, "p_champ_model": reach_m[pos[champ], K],
                "p_champ_elo": reach_elo[pos[champ], K], "favorite": fav,
                # puesto del campeón en la lista de favoritos del modelo (1 = favorito)
                "champ_rank_model": int(1 + (reach_m[:, K] > reach_m[pos[champ], K]).sum()),
                "p_fav_model": reach_m[pos[fav], K], "fav_won": int(fav == champ),
            })
            # celdas jugador x ronda (contadas desde el final: 0 = campeón, 1 = final, 2 = semis, 3 = cuartos)
            for j in range(1, K + 1):
                for i in range(n):
                    out[a][1].append((K - j, level, reach_m[i, j], reach_elo[i, j], actual[i, j]))

            if show:
                names = {}
                for side in ("winner", "loser"):
                    names.update(dict(zip(g[f"{side}_id"], g[f"{side}_name"])))
                rn = round_names(len(slots_ids))
                t = pd.DataFrame(reach_m[:, 1:] * 100, columns=rn[1:])
                t.insert(0, "Jugador", [names[p] for p in players])
                t["Elo: campeón"] = reach_elo[:, K] * 100
                t["Real"] = [rn[reached.get(p, 0)] for p in players]
                t = t.sort_values("Campeón", ascending=False).reset_index(drop=True)
                t.index += 1
                print(f"\n{info['tourney_name']} {date.date()} | {surface}, al mejor de {best_of} | {n} jugadores"
                      f" | calibración a = {a:.3f}")
                print("Probabilidad (%) que daba el modelo ANTES del torneo, y ronda a la que llegó cada uno:\n")
                with pd.option_context("display.width", 220, "display.max_columns", 30):
                    print(t.head(20).round(1).to_string())
                print(f"\nCampeón: {names[champ]} — el modelo le daba {reach_m[pos[champ], K] * 100:.1f} % "
                      f"(solo Elo: {reach_elo[pos[champ], K] * 100:.1f} %)")

    info = {"a_match": a_match, "model_name": model_name, "sims": sims}
    return out, skipped, info


def summary(res, cells):
    """Métricas principales para comparar valores de calibración."""
    eps = 1e-4
    c = pd.DataFrame(cells, columns=["from_end", "level", "p_model", "p_elo", "actual"])
    row = {"ll_champ": float(-np.log(res["p_champ_model"].clip(lower=eps)).mean())}
    for fe, name in [(0, "champ"), (1, "final"), (2, "sf")]:
        cc = c[c["from_end"] == fe]
        row[f"brier_{name}"] = float(((cc["p_model"] - cc["actual"]) ** 2).mean())
    top = c[(c["from_end"] == 0) & (c["p_model"] >= 0.35)]
    row["top_n"] = len(top)
    row["top_pred"] = float(top["p_model"].mean()) if len(top) else float("nan")
    row["top_real"] = float(top["actual"].mean()) if len(top) else float("nan")
    return row


def tune(a_grid, sims):
    """Elige la calibración a nivel de torneo con los torneos de VALIDACIÓN (2022-2023)."""
    out, skipped, info = run_backtest("val", a_grid, sims)
    lines = []

    def say(text=""):
        print(text)
        lines.append(str(text))

    n_t = len(out[a_grid[0]][0])
    say(f"Calibración para torneos elegida con los torneos de VALIDACIÓN ({TRAIN_END_YEAR + 1}-{VAL_END_YEAR})")
    say(f"Modelo entrenado solo con 2005-{TRAIN_END_YEAR}, sin calibrar | {n_t} torneos | {sims:,} simulaciones")
    say(f"Calibración por partido que usa el modelo final (ajustada partido a partido en validación): "
        f"a = {load_model('test')[1]:.3f}\n")
    say(f"  {'a':>6s} {'log-loss campeón':>17s} {'Brier campeón':>14s} {'Brier final':>12s} {'Brier semis':>12s}"
        f"   favoritos con >= 35 %: casos / predicho / real")
    rows = {}
    for a in a_grid:
        s = summary(pd.DataFrame(out[a][0]), out[a][1])
        rows[a] = s
        say(f"  {a:6.3f} {s['ll_champ']:17.4f} {s['brier_champ']:14.5f} {s['brier_final']:12.5f} {s['brier_sf']:12.5f}"
            f"   {s['top_n']:5d} / {s['top_pred'] * 100:5.1f} % / {s['top_real'] * 100:5.1f} %")
    best = min(a_grid, key=lambda a: rows[a]["ll_champ"])
    say(f"\n  Mejor a para torneos (menor log-loss del campeón): {best:.3f}")
    with open("backtest_tune_val.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("Guardado: backtest_tune_val.txt")
    return best


def main():
    ap = argparse.ArgumentParser(description="Backtest del simulador de torneos")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="test = 2024 en adelante (modelo 2005-2023); val = 2022-2023 (modelo 2005-2021)")
    ap.add_argument("--a", type=float, help="calibración para torneos (por defecto, la de partidos)")
    ap.add_argument("--compare-a", type=float, help="comparar la calibración principal con este otro valor de a")
    ap.add_argument("--tune-a", action="store_true", help="elegir la calibración de torneos en validación")
    ap.add_argument("--show", help="mostrar el detalle de un torneo (texto del nombre, p. ej. 'Roland Garros')")
    ap.add_argument("--year", type=int, help="año del torneo de --show")
    ap.add_argument("--sims", type=int, default=SIMS)
    args = ap.parse_args()

    if args.tune_a:
        tune([round(float(a), 3) for a in np.arange(0.60, 1.0001, 0.05)], args.sims)
        return

    _, a_match, _, _ = load_model(args.split)
    a_main = args.a if args.a is not None else a_match
    a_values = [a_main] + ([args.compare_a] if args.compare_a is not None else [])
    out, skipped, info = run_backtest(args.split, a_values, args.sims, args.show, args.year)
    if args.show:
        return

    if args.compare_a is not None:
        print(f"Comparación en {args.split}: a = {a_main:.3f} frente a a = {args.compare_a:.3f}\n")
        res_a, res_b = pd.DataFrame(out[a_main][0]), pd.DataFrame(out[args.compare_a][0])
        for a, r, cl in [(a_main, res_a, out[a_main][1]), (args.compare_a, res_b, out[args.compare_a][1])]:
            s = summary(r, cl)
            print(f"  a = {a:.3f}: log-loss campeón {s['ll_champ']:.4f} | Brier campeón {s['brier_champ']:.5f} "
                  f"final {s['brier_final']:.5f} semis {s['brier_sf']:.5f} | favoritos >= 35 %: "
                  f"{s['top_n']} casos, pred. {s['top_pred'] * 100:.1f} % / real {s['top_real'] * 100:.1f} %")
        diff = np.log(res_b["p_champ_model"].clip(lower=1e-4)) - np.log(res_a["p_champ_model"].clip(lower=1e-4))
        half = 1.96 * diff.std(ddof=1) / math.sqrt(len(diff))
        print(f"\n  Mejora de a = {args.compare_a:.3f} frente a a = {a_main:.3f} (log-loss del campeón, > 0 = mejor): "
              f"{diff.mean():+.4f} [{diff.mean() - half:+.4f}, {diff.mean() + half:+.4f}] IC 95 %")
        return

    period = "2024 en adelante" if args.split == "test" else f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}"
    trained = f"2005-{VAL_END_YEAR}" if args.split == "test" else f"2005-{TRAIN_END_YEAR}"
    header = (f"Torneos simulados: {{n}} ({period}) | descartados: {len(skipped)} | "
              f"{args.sims:,} simulaciones por torneo\n"
              f"Modelo: entrenado solo con {trained} (no ha visto ningún torneo de este informe) | "
              f"calibración a = {a_main:.3f}\n")
    suffix = "" if args.split == "test" else "_val"
    report(pd.DataFrame(out[a_main][0]), out[a_main][1], skipped, header, suffix)


def report(res, cells, skipped, header, suffix=""):
    lines = []

    def say(text=""):
        print(text)
        lines.append(str(text))

    eps = 1e-4
    say(header.format(n=len(res)))

    # ---------- 1) Probabilidad dada al campeón real ----------
    say("1) PROBABILIDAD QUE SE LE DABA AL CAMPEÓN REAL (antes del torneo)")
    say("   Log-loss del campeón = -log(prob. del campeón). Menor = mejor. 'Azar' = 1 / nº de jugadores.\n")
    say(f"   {'grupo':14s} {'torneos':>8s} {'modelo':>8s} {'solo Elo':>9s} {'azar':>7s}   "
        f"{'prob. media campeón (modelo / Elo)':>36s}")
    groups = [("Todos", res)] + [(lv, res[res["level"] == lv]) for lv in
                                 ["Grand Slam", "Masters 1000", "ATP 250/500", "Challenger", "Otros"]]
    for name, gdf in groups:
        if gdf.empty:
            continue
        ll_m = -np.log(gdf["p_champ_model"].clip(lower=eps)).mean()
        ll_e = -np.log(gdf["p_champ_elo"].clip(lower=eps)).mean()
        ll_u = np.log(gdf["players"]).mean()
        say(f"   {name:14s} {len(gdf):8d} {ll_m:8.3f} {ll_e:9.3f} {ll_u:7.3f}   "
            f"{gdf['p_champ_model'].mean() * 100:17.1f} % / {gdf['p_champ_elo'].mean() * 100:.1f} %")
    better = (res["p_champ_model"] > res["p_champ_elo"]).mean()
    say(f"\n   El modelo dio más probabilidad al campeón que el Elo en el {better * 100:.0f} % de los torneos.")
    # diferencia emparejada torneo a torneo, con intervalo de confianza del 95 %
    say("   Mejora del modelo frente al Elo (log-loss Elo - modelo; > 0 = el modelo es mejor), IC 95 %:")
    for name, gdf in groups:
        if len(gdf) < 5:
            continue
        diff = np.log(gdf["p_champ_model"].clip(lower=eps)) - np.log(gdf["p_champ_elo"].clip(lower=eps))
        half = 1.96 * diff.std(ddof=1) / math.sqrt(len(diff))
        verdict = "significativa" if diff.mean() - half > 0 else ("peor" if diff.mean() + half < 0 else "no concluyente")
        say(f"     {name:14s} {diff.mean():+.3f}  [{diff.mean() - half:+.3f}, {diff.mean() + half:+.3f}]  {verdict}")

    # ---------- 2) Favoritos ----------
    say("\n2) EL FAVORITO DEL MODELO, ¿GANÓ?")
    for name, gdf in groups:
        if gdf.empty:
            continue
        say(f"   {name:14s} ganó el {gdf['fav_won'].mean() * 100:5.1f} %  "
            f"(el modelo le daba de media un {gdf['p_fav_model'].mean() * 100:.1f} %) | campeón entre los "
            f"2 favoritos: {(gdf['champ_rank_model'] <= 2).mean() * 100:5.1f} %, entre los 4: "
            f"{(gdf['champ_rank_model'] <= 4).mean() * 100:5.1f} %")

    # ---------- 3) Calibración ----------
    c = pd.DataFrame(cells, columns=["from_end", "level", "p_model", "p_elo", "actual"])
    bins = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0001]
    for from_end, label in [(0, "SER CAMPEÓN"), (1, "LLEGAR A LA FINAL"), (2, "LLEGAR A SEMIFINALES")]:
        cc = c[c["from_end"] == from_end]
        say(f"\n3) CALIBRACIÓN - probabilidad de {label} (cada jugador de cada torneo)")
        say(f"   {'predicho':>12s} {'casos':>7s} {'pred. medio':>12s} {'real':>7s}")
        for a, b in zip(bins[:-1], bins[1:]):
            mk = (cc["p_model"] >= a) & (cc["p_model"] < b)
            if mk.sum():
                say(f"   {f'{a * 100:.0f}-{min(b, 1) * 100:.0f} %':>12s} {mk.sum():7d} "
                    f"{cc.loc[mk, 'p_model'].mean() * 100:11.1f}% {cc.loc[mk, 'actual'].mean() * 100:6.1f}%")
        bm = ((cc["p_model"] - cc["actual"]) ** 2).mean()
        be = ((cc["p_elo"] - cc["actual"]) ** 2).mean()
        say(f"   Brier: modelo {bm:.5f} | solo Elo {be:.5f} | mejora {100 * (be - bm) / be:+.1f} %")

    # ---------- 4) Grand Slams ----------
    gs = res[res["level"] == "Grand Slam"].sort_values("date")
    if len(gs):
        say("\n4) GRAND SLAMS")
        for _, r in gs.iterrows():
            say(f"   {str(r['date'])} {r['tourney_name']:18s} campeón {r['p_champ_model'] * 100:5.1f} % "
                f"(Elo {r['p_champ_elo'] * 100:5.1f} %) | el campeón era el favorito nº {r['champ_rank_model']}")

    if skipped:
        say(f"\nTorneos descartados ({len(skipped)}): eliminatoria incompleta o formato liguilla")
        for nm, d, why in skipped:
            say(f"   {d} {nm}: {why}")
    fb = int(res["fallback_profiles"].sum())
    say(f"\nJugadores que solo jugaron walkovers (estado tomado de su torneo anterior): {fb}")

    res.to_csv(f"backtest_tournaments{suffix}.csv", index=False)
    say(f"\nGuardado: backtest_tournaments{suffix}.csv y backtest_report{suffix}.txt")
    with open(f"backtest_report{suffix}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
