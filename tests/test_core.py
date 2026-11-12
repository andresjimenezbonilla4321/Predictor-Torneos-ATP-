"""
Tests rápidos (no necesitan los datos descargados).
Ejecutar:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (FEATURES, START_RATING, CalibratedModel, SymmetricModel, accuracy, calibration_table,  # noqa: E402
                    expected, history_state, inactivity_shrink, logloss, make_features, serve_state,
                    total_games)
from simulate_tournament import BYE, bracket_order, round_names, simulate  # noqa: E402


# ---------------- marcadores ----------------
@pytest.mark.parametrize("score, games", [
    ("6-4 6-3", 19),
    ("7-6(5) 6-7(3) 6-4", 36),
    ("6-4 3-6 [10-8]", 19),        # el super tie-break no cuenta como juegos
    ("6-2 3-1 RET", 12),
    ("W/O", 0),
    (None, 0),
])
def test_total_games(score, games):
    assert total_games(score) == games


# ---------------- Elo ----------------
def test_expected_is_symmetric():
    assert expected(1500, 1500) == pytest.approx(0.5)
    assert expected(1800, 1600) + expected(1600, 1800) == pytest.approx(1.0)
    assert expected(1900, 1500) == pytest.approx(1 / (1 + 10 ** (-400 / 400)))


def test_inactivity_shrink():
    assert inactivity_shrink(2000, 30, 120, 0.2) == 2000              # ausencia corta: nada
    assert inactivity_shrink(2000, 365, 120, 0.2) == pytest.approx(2000 - 0.2 * 500)
    assert inactivity_shrink(2000, 2000, 120, 0.2) == pytest.approx(1900)   # tope de 1 año
    assert inactivity_shrink(1300, 365, 120, 0.2) > 1300              # los de abajo suben hacia 1500
    assert inactivity_shrink(START_RATING, 365, 120, 0.2) == START_RATING


# ---------------- historial ----------------
def test_history_state():
    recent = [(100, 20), (110, 25), (120, 30)]   # (día, juegos)
    form, rest, layoff, g14 = history_state(recent, [1, 1, 0], d=120)
    assert form == pytest.approx((2 + 2.5) / (3 + 5))
    assert rest == 10           # el partido del día 120 es del mismo torneo: no cuenta como descanso
    assert layoff == 10
    assert g14 == 25 + 30       # día 110 y 120 están dentro de 14 días


def test_history_state_new_player():
    form, rest, layoff, g14 = history_state([], [], d=500)
    assert form == 0.5 and g14 == 0


def test_serve_state_shrinks_to_prior():
    prior = {"spw": 0.62, "rpw": 0.38, "ace": 0.07, "df": 0.04, "fin": 0.6}
    assert serve_state([], prior)[:5] == [0.62, 0.38, 0.07, 0.04, 0.6]
    # un partido perfecto al saque no debería dar 100 %: se suaviza hacia la media
    spw = serve_state([(50, 50, 50, 10, 5, 0, 30)], prior)[0]
    assert 0.62 < spw < 0.75


# ---------------- características ----------------
def _player(**kw):
    base = dict(elo=1800.0, elo_surf=1750.0, trend=30.0, rank=20.0, points=2000.0, age=25.0, ht=185.0, hand="R", n=100.0, form=0.6,
                rest=7.0, layoff=7.0, g14=40.0, h2h=2.0, spw=0.65, rpw=0.38, ace=0.08, df=0.03, fin=0.6)
    base.update(kw)
    return {k: np.array([v], dtype=object if k == "hand" else float) for k, v in base.items()}


def test_make_features_antisymmetric():
    A = _player()
    B = _player(elo=1700.0, elo_surf=np.nan, trend=-40.0, rank=np.nan, points=np.nan, age=np.nan, hand="L",
                h2h=5.0, layoff=200.0)
    fab, fba = make_features(A, B, [5]), make_features(B, A, [5])
    assert list(fab) == FEATURES
    for k in FEATURES:
        assert fab[k][0] == pytest.approx(-fba[k][0]), k
        assert np.isfinite(fab[k][0]), k
    assert fab["elo_diff_bo5"][0] == fab["elo_diff"][0]
    assert make_features(A, B, [3])["elo_diff_bo5"][0] == 0


# ---------------- modelos ----------------
def _toy_dataset(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    p = 1 / (1 + np.exp(-(X[:, 0] * 1.5 + X[:, 1] * 0.5)))
    df = pd.DataFrame(X, columns=FEATURES)
    df["target"] = (rng.random(n) < p).astype(int)
    return df


@pytest.mark.parametrize("kind, params", [("logreg", None), ("hgb", (3, 0.1, 50, 20))])
def test_symmetric_model(kind, params):
    df = _toy_dataset()
    m = SymmetricModel("t", kind, params).fit(df)
    p = m.predict(df)
    p_flip = m.predict(-df[FEATURES])
    assert np.allclose(p + p_flip, 1.0)
    assert accuracy(df["target"], p) > 0.65


def test_calibration_keeps_symmetry_and_helps():
    y = np.array([1, 0] * 500)
    p = np.where(y == 1, 0.9, 0.3)              # demasiado confiado en los aciertos
    a = CalibratedModel.fit_scale(y, p)
    assert logloss(y, CalibratedModel.apply(p, a)) <= logloss(y, p)
    assert CalibratedModel.apply(0.7, 0.8) + CalibratedModel.apply(0.3, 0.8) == pytest.approx(1.0)
    assert len(calibration_table(y, p)) > 0


# ---------------- simulador ----------------
def test_bracket_order():
    assert bracket_order(2) == [1, 2]
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    order = bracket_order(32)
    assert sorted(order) == list(range(1, 33))
    assert order.index(1) < 16 <= order.index(2)          # 1 y 2 en mitades distintas


def test_round_names():
    assert round_names(16) == ["R16", "QF", "SF", "F", "Campeón"]
    assert round_names(2) == ["F", "Campeón"]


def test_simulate_deterministic_and_byes():
    # el jugador 0 siempre gana, el 1 siempre gana a 2
    prob = np.array([[0.5, 1.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.5]])
    slots = np.array([0, BYE, 1, 2])
    reach = simulate(prob, slots, sims=1000, rng=np.random.default_rng(0)) / 1000
    assert reach[0].tolist() == [1, 1, 1]       # pasa el BYE, gana la final
    assert reach[1].tolist() == [1, 1, 0]
    assert reach[2].tolist() == [1, 0, 0]


def test_simulate_probabilities_sum():
    n = 8
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.2, 0.8, size=(n, n))
    prob = np.triu(raw, 1) + np.tril(1 - raw.T, -1) + np.eye(n) * 0.5
    reach = simulate(prob, np.arange(n), sims=4000, rng=rng) / 4000
    assert reach[:, -1].sum() == pytest.approx(1.0)     # siempre hay exactamente un campeón
    assert reach[:, 1].sum() == pytest.approx(n / 2)


# ---------------- reconstrucción de cuadros (backtest) ----------------
from backtest_tournaments import BracketError, build_bracket  # noqa: E402


def test_build_bracket_full_draw():
    rows = [("QF", 1, 8), ("QF", 4, 5), ("QF", 2, 7), ("QF", 3, 6),
            ("SF", 1, 4), ("SF", 3, 2), ("F", 3, 1)]
    slots, rounds, reached = build_bracket(rows)
    assert rounds == ["QF", "SF", "F"]
    assert slots == [3, 6, 2, 7, 1, 8, 4, 5]          # 3 y 6 se cruzan en QF, su ganador con el de 2-7...
    pairs = {frozenset(slots[i:i + 2]) for i in range(0, 8, 2)}
    assert pairs == {frozenset(p) for p in [(1, 8), (4, 5), (2, 7), (3, 6)]}
    assert reached == {3: 3, 1: 2, 2: 1, 4: 1, 8: 0, 5: 0, 7: 0, 6: 0}


def test_build_bracket_with_byes():
    # 6 jugadores: 1 y 2 empiezan en semis (BYE en cuartos)
    rows = [("QF", 3, 4), ("QF", 5, 6), ("SF", 1, 3), ("SF", 2, 5), ("F", 2, 1)]
    slots, rounds, reached = build_bracket(rows)
    assert len(slots) == 8 and slots.count(BYE) == 2
    assert sorted(s for s in slots if s != BYE) == [1, 2, 3, 4, 5, 6]
    assert reached[2] == 3 and reached[1] == 2 and reached[4] == 0
    # al simular con probabilidad 1 para el de menor id, gana el 1
    players = [s for s in slots if s != BYE]
    pos = {p: i for i, p in enumerate(players)}
    prob = np.array([[0.5 if a == b else float(a < b) for b in players] for a in players])
    sl = np.array([BYE if s == BYE else pos[s] for s in slots])
    reach = simulate(prob, sl, sims=200, rng=np.random.default_rng(0)) / 200
    assert reach[pos[1], -1] == 1.0


@pytest.mark.parametrize("rows", [
    [("RR", 1, 2), ("F", 1, 3)],                    # liguilla
    [("QF", 1, 2), ("F", 1, 3)],                    # faltan las semifinales
    [("SF", 1, 2), ("SF", 1, 3), ("F", 1, 4)],      # el mismo jugador gana dos semis
])
def test_build_bracket_rejects_bad_structures(rows):
    with pytest.raises(BracketError):
        build_bracket(rows)


# ---------------- Elo mejorado y cuotas ----------------
from common import games_split  # noqa: E402
from compare_odds import blend, odds_key, player_key  # noqa: E402
from compute_elo import elo_trend  # noqa: E402


def test_games_split():
    assert games_split("6-1 6-1") == (12, 2)
    assert games_split("7-6(5) 6-7(3) 7-6(8)") == (20, 19)
    assert games_split("6-4 3-6 [10-8]") == (9, 10)       # el ganador puede tener menos juegos
    assert games_split("W/O") == (0, 0)


def test_elo_trend():
    hist_d = {1: [0, 50, 100, 150]}
    hist_r = {1: [1500, 1550, 1600, 1650]}
    assert elo_trend(hist_d, hist_r, 1, 1700, d=200, days=90) == 1700 - 1600   # Elo del día 100 (<= 110)
    assert elo_trend(hist_d, hist_r, 1, 1700, d=60, days=90) == 0.0           # sin historial tan antiguo
    assert elo_trend(hist_d, hist_r, 2, 1700, d=200, days=90) == 0.0          # jugador sin historial


@pytest.mark.parametrize("name, key", [
    ("Sinner J.", "sinner_j"),
    ("De Minaur A.", "deminaur_a"),
    ("Auger-Aliassime F.", "augeraliassime_f"),
    ("Etcheverry T.M.", "etcheverry_t"),
])
def test_odds_key(name, key):
    assert odds_key(name) == key


def test_player_key_matches_odds_key():
    assert player_key("Alex", "De Minaur") == odds_key("De Minaur A.")
    assert player_key("Félix", "Auger Aliassime") == odds_key("Auger-Aliassime F.")


def test_blend_extremes():
    pm, pb = np.array([0.7]), np.array([0.6])
    assert blend(pm, pb, 1.0)[0] == pytest.approx(0.7)
    assert blend(pm, pb, 0.0)[0] == pytest.approx(0.6)
    assert 0.6 < blend(pm, pb, 0.5)[0] < 0.7
