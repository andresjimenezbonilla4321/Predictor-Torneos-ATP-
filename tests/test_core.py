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
    base = dict(elo=1800.0, elo_surf=1750.0, rank=20.0, age=25.0, ht=185.0, hand="R", n=100.0, form=0.6,
                rest=7.0, layoff=7.0, g14=40.0, h2h=2.0, spw=0.65, rpw=0.38, ace=0.08, df=0.03, fin=0.6)
    base.update(kw)
    return {k: np.array([v], dtype=object if k == "hand" else float) for k, v in base.items()}


def test_make_features_antisymmetric():
    A = _player()
    B = _player(elo=1700.0, elo_surf=np.nan, rank=np.nan, age=np.nan, hand="L", h2h=5.0, layoff=200.0)
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
