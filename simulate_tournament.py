"""
Simulador Monte Carlo de un torneo ATP (cuadro de eliminación directa).

Con las probabilidades del modelo para cada posible partido, se juega el torneo
miles de veces y se cuenta en qué ronda cae cada jugador. El resultado es la
probabilidad de que cada uno llegue a cada ronda y de que gane el título.

Uso:
  # cuadro en un archivo de texto (un jugador por línea, en el orden del cuadro, BYE permitido)
  python simulate_tournament.py --draw draws/ejemplo_16_clay.txt --surface Clay

  # lista de jugadores sin orden: se colocan como cabezas de serie según su Elo
  python simulate_tournament.py --draw draws/ejemplo_16_clay.txt --seed-by-elo --surface Clay

  # los N mejores jugadores activos por Elo, cuadro con cabezas de serie
  python simulate_tournament.py --top 32 --surface Grass --best-of 5

Opciones: --sims 20000  --date AAAA-MM-DD  --out resultado.csv  --seed 0

Nota: se usa el estado de cada jugador ANTES del torneo para todas las rondas
(no se modela el cansancio que se acumula dentro del propio torneo).
"""
import argparse
import math

import numpy as np
import pandas as pd

from predict import Predictor

BYE = -1


def bracket_order(n):
    """
    Orden estándar de cabezas de serie en un cuadro de n (potencia de 2), con 1-based seeds:
    n=8 -> [1, 8, 4, 5, 2, 7, 3, 6]  (1 y 2 solo pueden cruzarse en la final).
    """
    order = [1]
    while len(order) < n:
        m = 2 * len(order) + 1
        order = [x for s in order for x in (s, m - s)]
    return order


def round_names(size):
    names = {2: "F", 4: "SF", 8: "QF"}
    rounds = []
    s = size
    while s >= 2:
        rounds.append(names.get(s, f"R{s}"))
        s //= 2
    return rounds + ["Campeón"]


def read_draw(path):
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                entries.append(line)
    return entries


def simulate(prob, slots, sims, rng):
    """
    prob : matriz (n, n) con P(i gana a j) entre los jugadores reales
    slots: array con el índice de jugador de cada posición del cuadro (BYE = -1)
    Devuelve reach (n, n_rondas+1): nº de simulaciones en que cada jugador alcanza cada ronda.
    """
    n = prob.shape[0]
    size = len(slots)
    n_rounds = int(math.log2(size))
    # matriz ampliada con el BYE en la última fila/columna: cualquiera gana a un BYE
    P = np.zeros((n + 1, n + 1))
    P[:n, :n] = prob
    P[:n, n] = 1.0
    P[n, :n] = 0.0
    P[n, n] = 1.0  # BYE vs BYE: "gana" un BYE (no debería ocurrir con cuadros sensatos)
    idx = np.where(slots == BYE, n, slots)

    alive = np.tile(idx, (sims, 1))                    # (sims, size)
    reach = np.zeros((n + 1, n_rounds + 1))
    np.add.at(reach[:, 0], idx, sims)
    for r in range(n_rounds):
        a, b = alive[:, 0::2], alive[:, 1::2]
        p = P[a, b]
        win_a = rng.random(p.shape) < p
        alive = np.where(win_a, a, b)
        np.add.at(reach[:, r + 1], alive.ravel(), 1)
    return reach[:n]


def top_players(pr, n, active_days=365):
    """Los n mejores jugadores activos (han jugado en el último año) por Elo actual."""
    last = pr.elo["last_day"]
    active = [p for p in pr.elo["gen_r"] if pr.day - last.get(p, -10**9) <= active_days]
    return sorted(active, key=lambda p: -pr.current_elo(p))[:n]


def simulate_draw(pr, entries, surface="Hard", best_of=3, sims=20000, seed_by_elo=False, is_main=1, seed=0):
    """
    Simula un cuadro. entries: lista de player_id (o BYE) en orden de cuadro, o sin orden si seed_by_elo.
    Devuelve un dict con la tabla de probabilidades (%), la primera ronda y el tamaño del cuadro.
    """
    players = [p for p in entries if p != BYE]
    if len(players) < 2:
        raise ValueError("Hacen falta al menos 2 jugadores.")
    if len(set(players)) != len(players):
        dup = {pr.name(p) for p in players if players.count(p) > 1}
        raise ValueError(f"Jugadores repetidos en el cuadro: {', '.join(dup)}")

    profiles = {p: pr.profile(p, surface) for p in players}

    if seed_by_elo:
        size = 2 ** math.ceil(math.log2(len(players)))
        ranked = sorted(players, key=lambda p: -profiles[p]["elo"])
        ranked += [BYE] * (size - len(ranked))        # los BYE son los "peores" cabezas: tocan a los mejores
        slots_ids = [ranked[s - 1] for s in bracket_order(size)]
    else:
        size = len(entries)
        if size & (size - 1):
            raise ValueError(f"El cuadro tiene {size} posiciones; debe ser potencia de 2 (añade BYE) "
                             "o colócalo por Elo para que se complete solo.")
        slots_ids = list(entries)

    pos = {p: i for i, p in enumerate(players)}
    slots = np.array([BYE if p == BYE else pos[p] for p in slots_ids])

    pairs = [(a, b) for a in players for b in players if a != b]
    probs = pr.pairs_proba(pairs, surface, best_of, profiles=profiles, is_main=is_main)
    n = len(players)
    prob = np.full((n, n), 0.5)
    for (a, b), p in zip(pairs, probs):
        prob[pos[a], pos[b]] = p

    reach = simulate(prob, slots, sims, np.random.default_rng(seed)) / sims
    names = round_names(size)
    table = pd.DataFrame(reach[:, 1:] * 100, columns=names[1:])
    table.insert(0, "Jugador", [pr.name(p) for p in players])
    table.insert(1, "Elo", [round(profiles[p]["elo"]) for p in players])
    table.insert(2, f"Elo {surface}", [round(profiles[p]["elo_surf"]) for p in players])
    table = table.sort_values("Campeón", ascending=False).reset_index(drop=True)
    table.index += 1

    first_round = []
    for i in range(0, size, 2):
        a, b = slots_ids[i], slots_ids[i + 1]
        pa = None if BYE in (a, b) else float(prob[pos[a], pos[b]])
        first_round.append(("BYE" if a == BYE else pr.name(a), "BYE" if b == BYE else pr.name(b), pa))
    return {"table": table, "first_round": first_round, "size": size, "n": n, "sims": sims}


def main():
    ap = argparse.ArgumentParser(description="Simulación Monte Carlo de un torneo ATP")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--draw", help="archivo con el cuadro (un jugador por línea; 'BYE' para huecos)")
    src.add_argument("--top", type=int, help="usar los N mejores jugadores activos por Elo")
    ap.add_argument("--seed-by-elo", action="store_true",
                    help="reordenar a los jugadores como cabezas de serie por Elo (con --top siempre)")
    ap.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--challenger", action="store_true", help="torneo Challenger (por defecto, circuito principal)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--date", help="fecha del torneo (AAAA-MM-DD). Por defecto: 1 semana tras el fin de los datos")
    ap.add_argument("--seed", type=int, default=0, help="semilla aleatoria (para repetir resultados)")
    ap.add_argument("--out", help="guardar la tabla de resultados en CSV")
    args = ap.parse_args()

    pr = Predictor(date=args.date)
    if args.top:
        entries, seed_by_elo = top_players(pr, args.top), True
    else:
        entries = [BYE if e.upper() == "BYE" else pr.find_player(e) for e in read_draw(args.draw)]
        seed_by_elo = args.seed_by_elo
    try:
        res = simulate_draw(pr, entries, args.surface, args.best_of, args.sims, seed_by_elo,
                            0 if args.challenger else 1, args.seed)
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"\nTorneo simulado: {res['n']} jugadores, cuadro de {res['size']} | {args.surface}, al mejor de "
          f"{args.best_of} | {args.sims:,} simulaciones | fecha {pr.date.date()}")
    print(f"Modelo: {pr.model_name}\n")
    print("Primera ronda:")
    for na, nb, pa in res["first_round"]:
        p = "" if pa is None else f"  ({pa * 100:.0f} % - {(1 - pa) * 100:.0f} %)"
        print(f"  {na:>26s}  vs  {nb:<26s}{p}")
    print("\nProbabilidad (%) de llegar a cada ronda:")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res["table"].round(1).to_string())
    margin = 100 * 1.96 * math.sqrt(0.25 / args.sims)
    print(f"\n(Error de la simulación: como mucho ±{margin:.1f} puntos. Sube --sims para afinar.)")

    if args.out:
        res["table"].to_csv(args.out, index_label="puesto")
        print(f"Guardado: {args.out}")


if __name__ == "__main__":
    main()
