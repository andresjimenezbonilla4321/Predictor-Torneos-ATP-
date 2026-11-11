"""
Simulador Monte Carlo de un torneo ATP (cuadro de eliminación directa).

Con las probabilidades del modelo para cada posible partido, se juega el torneo
miles de veces y se cuenta en qué ronda cae cada jugador. El resultado es la
probabilidad de que cada uno llegue a cada ronda y de que gane el título.

Uso:
  # cuadro en un archivo de texto (un jugador por línea, en el orden del cuadro, BYE permitido)
  python simulate_tournament.py --draw draws/ejemplo_masters_clay.txt --surface Clay

  # lista de jugadores sin orden: se colocan como cabezas de serie según su Elo
  python simulate_tournament.py --draw draws/ejemplo_masters_clay.txt --seed-by-elo --surface Clay

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


def main():
    ap = argparse.ArgumentParser(description="Simulación Monte Carlo de un torneo ATP")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--draw", help="archivo con el cuadro (un jugador por línea; 'BYE' para huecos)")
    src.add_argument("--top", type=int, help="usar los N mejores jugadores activos por Elo")
    ap.add_argument("--seed-by-elo", action="store_true",
                    help="reordenar a los jugadores como cabezas de serie por Elo (con --top siempre)")
    ap.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--date", help="fecha del torneo (AAAA-MM-DD). Por defecto: 1 semana tras el fin de los datos")
    ap.add_argument("--seed", type=int, default=0, help="semilla aleatoria (para repetir resultados)")
    ap.add_argument("--out", help="guardar la tabla de resultados en CSV")
    args = ap.parse_args()

    pr = Predictor(date=args.date)

    # ---------- 1) Jugadores ----------
    if args.top:
        elo, last = pr.elo["gen_r"], pr.elo["last_day"]
        active = [p for p in elo if pr.day - last.get(p, -10**9) <= 365]
        prof = {p: pr.profile(p, args.surface) for p in active}
        ids = sorted(active, key=lambda p: -prof[p]["elo"])[:args.top]
        entries, seed_by_elo = ids, True
    else:
        raw = read_draw(args.draw)
        entries = [BYE if e.upper() == "BYE" else pr.find_player(e) for e in raw]
        seed_by_elo = args.seed_by_elo

    players = [p for p in entries if p != BYE]
    if len(set(players)) != len(players):
        dup = {pr.name(p) for p in players if players.count(p) > 1}
        raise SystemExit(f"Jugadores repetidos en el cuadro: {', '.join(dup)}")

    profiles = {p: pr.profile(p, args.surface) for p in players}

    # ---------- 2) Cuadro ----------
    if seed_by_elo:
        size = 2 ** math.ceil(math.log2(max(len(players), 2)))
        ranked = sorted(players, key=lambda p: -profiles[p]["elo"])
        ranked += [BYE] * (size - len(ranked))        # los BYE son los "peores" cabezas: tocan a los mejores
        slots_ids = [ranked[s - 1] for s in bracket_order(size)]
    else:
        size = len(entries)
        if size < 2 or size & (size - 1):
            raise SystemExit(f"El cuadro tiene {size} posiciones; debe ser potencia de 2 (añade BYE) "
                             "o usa --seed-by-elo para que se complete solo.")
        slots_ids = entries

    pos = {p: i for i, p in enumerate(players)}
    slots = np.array([BYE if p == BYE else pos[p] for p in slots_ids])

    # ---------- 3) Probabilidades de todos los cruces posibles ----------
    pairs = [(a, b) for a in players for b in players if a != b]
    probs = pr.pairs_proba(pairs, args.surface, args.best_of, profiles=profiles)
    n = len(players)
    prob = np.full((n, n), 0.5)
    for (a, b), p in zip(pairs, probs):
        prob[pos[a], pos[b]] = p

    # ---------- 4) Simulación ----------
    rng = np.random.default_rng(args.seed)
    reach = simulate(prob, slots, args.sims, rng) / args.sims
    names = round_names(size)

    table = pd.DataFrame(reach[:, 1:] * 100, columns=names[1:])
    table.insert(0, "Jugador", [pr.name(p) for p in players])
    table.insert(1, "Elo", [round(profiles[p]["elo"]) for p in players])
    table.insert(2, f"Elo {args.surface}", [round(profiles[p]["elo_surf"]) for p in players])
    table = table.sort_values("Campeón", ascending=False).reset_index(drop=True)
    table.index += 1

    print(f"\nTorneo simulado: {n} jugadores, cuadro de {size} | {args.surface}, al mejor de {args.best_of} | "
          f"{args.sims:,} simulaciones | fecha {pr.date.date()}")
    print(f"Modelo: {pr.model_name}\n")
    print("Primera ronda:")
    for i in range(0, size, 2):
        a, b = slots_ids[i], slots_ids[i + 1]
        na = "BYE" if a == BYE else pr.name(a)
        nb = "BYE" if b == BYE else pr.name(b)
        p = "" if BYE in (a, b) else f"  ({prob[pos[a], pos[b]] * 100:.0f} % - {prob[pos[b], pos[a]] * 100:.0f} %)"
        print(f"  {na:>26s}  vs  {nb:<26s}{p}")
    print("\nProbabilidad (%) de llegar a cada ronda:")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(1).to_string())
    margin = 100 * 1.96 * math.sqrt(0.25 / args.sims)
    print(f"\n(Error de la simulación: como mucho ±{margin:.1f} puntos. Sube --sims para afinar.)")

    if args.out:
        table.to_csv(args.out, index_label="puesto")
        print(f"Guardado: {args.out}")


if __name__ == "__main__":
    main()
