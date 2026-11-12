"""
Funciones y configuración compartidas por todo el proyecto.

Todo lo que se usa en más de un script vive aquí, para que el pipeline de
entrenamiento y la predicción de partidos futuros (predict.py) calculen las
características EXACTAMENTE igual.
"""
import pickle
import re
from pathlib import Path

import numpy as np

# ------------------------- CONFIGURACIÓN GLOBAL -------------------------
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"      # estado final de cada jugador (para predecir el futuro)
MODELS_DIR = ROOT / "models"    # modelos entrenados

START_RATING = 1500.0
FIRST_YEAR = 2005        # años anteriores = "calentamiento" del Elo y del historial
TRAIN_END_YEAR = 2021    # train: FIRST_YEAR..2021
VAL_END_YEAR = 2023      # val: 2022-2023 | test: 2024 en adelante
UNRANKED_VALUE = 1500    # ranking que se asigna a quien no tiene ranking ATP

# historial (add_features.py)
FORM_WINDOW = 10         # forma = % de victorias en los últimos N partidos
FORM_PRIOR = 5           # "partidos imaginarios" al 50% para no fiarse de 1-2 partidos
FATIGUE_DAYS = 14        # ventana para contar juegos jugados recientemente
REST_CAP = 30            # días de descanso máximos que consideramos
LAYOFF_CAP = 365         # días de inactividad máximos que consideramos

# saque y resto (add_serve_stats.py)
SERVE_WINDOW = 20        # últimos N partidos con estadísticas
SHRINK_PTS = 300         # "puntos imaginarios" con la media del circuito (~4 partidos)

# Características que usa el modelo (todas son diferencias A - B: cambian de signo
# si se intercambian A y B, lo que permite un modelo simétrico)
FEATURES = ["elo_diff", "elo_surf_diff", "elo_diff_bo5", "elo_diff_main", "elo_trend_diff",
            "log_rank_diff", "log_points_diff", "age_diff", "ht_diff",
            "lefty_diff", "exp_diff", "form_diff", "rest_diff", "layoff_diff", "games14_diff",
            "h2h_diff", "spw_diff", "rpw_diff", "ace_diff", "df_diff", "fin_diff"]
# -----------------------------------------------------------------------

DAY0 = np.datetime64("2000-01-01")


def to_day(dates):
    """Fecha(s) -> nº de días desde 2000-01-01 (entero)."""
    return (np.asarray(dates, dtype="datetime64[D]") - DAY0).astype(int)


def total_games(score):
    """Juegos totales de un marcador tipo '6-4 3-6 7-6(5) [10-8]' (el super tie-break no cuenta)."""
    s = re.sub(r"\[.*?\]", "", str(score))
    return sum(int(a) + int(b) for a, b in re.findall(r"(\d+)-(\d+)", s))


def games_split(score):
    """(juegos del ganador, juegos del perdedor) de un marcador; el ganador va siempre primero."""
    s = re.sub(r"\[.*?\]", "", str(score))
    pairs = re.findall(r"(\d+)-(\d+)", s)
    return sum(int(a) for a, _ in pairs), sum(int(b) for _, b in pairs)


# ------------------------------- Elo -------------------------------
def expected(r_a, r_b):
    """Probabilidad de que A gane a B según la diferencia de Elo."""
    return 1.0 / (1.0 + 10 ** ((np.asarray(r_b) - np.asarray(r_a)) / 400.0))


def inactivity_shrink(rating, gap_days, min_days, shrink):
    """
    Tras una ausencia larga el Elo se acerca a la media (1500): el jugador vuelve
    con menos ritmo y su nivel es más incierto.
    """
    if shrink <= 0 or gap_days <= min_days:
        return rating
    frac = shrink * min(gap_days / 365.0, 1.0)
    return rating - frac * (rating - START_RATING)


# ---------------------- Historial (forma, descanso) ----------------------
def history_state(recent, results, d):
    """
    recent : lista/deque de (día, juegos) de los partidos del jugador (más antiguos primero)
    results: lista/deque con sus últimos resultados (1 = victoria, 0 = derrota)
    d      : día del partido a predecir
    Devuelve (forma, descanso, inactividad, juegos_en_14_días).
    """
    g14 = sum(g for (dd, g) in recent if 0 <= d - dd < FATIGUE_DAYS)   # incluye rondas previas del torneo
    prev = [dd for (dd, g) in recent if dd < d]                          # torneos anteriores
    last = max(prev) if prev else None
    rest = min(d - last, REST_CAP) if last is not None else REST_CAP
    layoff = min(d - last, LAYOFF_CAP) if last is not None else LAYOFF_CAP
    form = (sum(results) + FORM_PRIOR * 0.5) / (len(results) + FORM_PRIOR)
    return form, rest, layoff, g14


# ---------------------- Saque y resto ----------------------
def serve_state(hist, prior):
    """
    hist : lista de tuplas (svpt, sv_ganados, pts_resto, resto_ganados, aces, dobles, primeros_in)
    prior: medias del circuito (dict con spw, rpw, ace, df, fin)
    Devuelve [spw, rpw, ace, df, fin, n_partidos] suavizados hacia la media.
    """
    if not len(hist):
        return [prior["spw"], prior["rpw"], prior["ace"], prior["df"], prior["fin"], 0]
    sv, won, rt, rwon, ace, df, fin = np.asarray(hist, dtype=float).sum(axis=0)
    k = SHRINK_PTS
    return [
        (won + k * prior["spw"]) / (sv + k),
        (rwon + k * prior["rpw"]) / (rt + k),
        (ace + k * prior["ace"]) / (sv + k),
        (df + k * prior["df"]) / (sv + k),
        (fin + k * prior["fin"]) / (sv + k),
        len(hist),
    ]


# ---------------------- Características del modelo ----------------------
def make_features(A, B, best_of, is_main=1):
    """
    A, B: dicts con los valores de cada jugador (arrays o escalares):
          elo, elo_surf, trend, rank, points, age, ht, hand, n, form, rest, layoff, g14, h2h,
          spw, rpw, ace, df, fin   (h2h = victorias de ese jugador sobre el otro)
    best_of: 3 o 5 | is_main: 1 = circuito principal, 0 = Challenger
    Devuelve un dict {feature: array} con las diferencias A - B.
    La usan build_model_dataset.py (histórico) y predict.py (futuro): mismas fórmulas.
    """
    f = lambda d, k: np.asarray(d[k], dtype=float)
    rank_a = np.nan_to_num(f(A, "rank"), nan=UNRANKED_VALUE)
    rank_b = np.nan_to_num(f(B, "rank"), nan=UNRANKED_VALUE)
    bo5 = (np.asarray(best_of, dtype=float) == 5).astype(float)
    main = np.asarray(is_main, dtype=float)
    elo_diff = f(A, "elo") - f(B, "elo")
    h2h_a, h2h_b = f(A, "h2h"), f(B, "h2h")
    out = {
        "elo_diff": elo_diff,
        # sin superficie conocida -> se usa el Elo general
        "elo_surf_diff": np.where(np.isnan(f(A, "elo_surf")), f(A, "elo"), f(A, "elo_surf"))
                         - np.where(np.isnan(f(B, "elo_surf")), f(B, "elo"), f(B, "elo_surf")),
        # al mejor de 5 sets el favorito gana más a menudo: el modelo aprende cuánto
        "elo_diff_bo5": elo_diff * bo5,
        # en el circuito principal el Elo puede pesar distinto que en los Challenger
        "elo_diff_main": elo_diff * main,
        # tendencia: cuánto ha subido (o bajado) el Elo de cada uno en los últimos meses
        "elo_trend_diff": f(A, "trend") - f(B, "trend"),
        # logaritmo: pasar del puesto 1 al 10 pesa más que del 101 al 110
        "log_rank_diff": np.log(rank_a) - np.log(rank_b),
        # puntos ATP (más finos que el puesto); sin ranking = 0 puntos
        "log_points_diff": np.log1p(np.nan_to_num(f(A, "points"), nan=0.0))
                           - np.log1p(np.nan_to_num(f(B, "points"), nan=0.0)),
        # alturas/edades que faltan -> diferencia 0 ("sin información")
        "age_diff": np.nan_to_num(f(A, "age") - f(B, "age"), nan=0.0),
        "ht_diff": np.nan_to_num(f(A, "ht") - f(B, "ht"), nan=0.0),
        "lefty_diff": (np.asarray(A["hand"]) == "L").astype(float) - (np.asarray(B["hand"]) == "L").astype(float),
        # experiencia: cuánto se puede fiar del Elo de cada uno
        "exp_diff": np.log1p(f(A, "n")) - np.log1p(f(B, "n")),
        "form_diff": f(A, "form") - f(B, "form"),
        "rest_diff": f(A, "rest") - f(B, "rest"),
        "layoff_diff": np.log1p(f(A, "layoff")) - np.log1p(f(B, "layoff")),
        "games14_diff": f(A, "g14") - f(B, "g14"),
        # cara a cara suavizado: (victorias de A - victorias de B) / (total + 2)
        "h2h_diff": (h2h_a - h2h_b) / (h2h_a + h2h_b + 2),
        "spw_diff": f(A, "spw") - f(B, "spw"),
        "rpw_diff": f(A, "rpw") - f(B, "rpw"),
        "ace_diff": f(A, "ace") - f(B, "ace"),
        "df_diff": f(A, "df") - f(B, "df"),
        "fin_diff": f(A, "fin") - f(B, "fin"),
    }
    assert list(out) == FEATURES
    return out


# ---------------------- Métricas ----------------------
def accuracy(y, p):
    return float(((np.asarray(p) > 0.5) == (np.asarray(y) == 1)).mean())


def logloss(y, p):
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2).mean())


def calibration_table(y, p, bins=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0001)):
    """
    Calibración: de los partidos en los que el favorito tenía p en [a, b),
    ¿cuántos ganó de verdad? Si el modelo está bien calibrado, 'real' ≈ 'predicho'.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    fav_p = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y == 1, y == 0)
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        mk = (fav_p >= a) & (fav_p < b)
        if mk.sum():
            rows.append((a, min(b, 1.0), int(mk.sum()), float(fav_p[mk].mean()), float(fav_won[mk].mean())))
    return rows


# ---------------------- Guardar / cargar estado ----------------------
def save_state(name, obj):
    STATE_DIR.mkdir(exist_ok=True)
    with open(STATE_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(name):
    path = STATE_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta antes el pipeline: python run_pipeline.py")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------- Modelo simétrico ----------------------
class SymmetricModel:
    """
    Envuelve un clasificador de sklearn para que P(A gana) = 1 - P(B gana) siempre:
    se promedia la predicción del partido y la del partido 'en espejo' (features con signo cambiado).
    kind = "logreg" (regresión logística sin intercepto) o "hgb" (gradient boosting con datos espejo).
    """

    def __init__(self, name, kind, params=None, features=FEATURES, tau=None):
        self.name, self.kind, self.params, self.features = name, kind, params, list(features)
        self.tau = tau   # años: si se indica, los partidos recientes pesan más al entrenar

    def _weights(self, df):
        if not self.tau or "match_date" not in df:
            return None
        years = df["match_date"].dt.year + df["match_date"].dt.dayofyear / 365.0
        return np.exp((years - years.max()).to_numpy() / self.tau)

    def fit(self, df):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = df[self.features].to_numpy(dtype=float)
        y = df["target"].to_numpy()
        w = self._weights(df)
        if self.kind == "logreg":
            self.model = make_pipeline(StandardScaler(with_mean=False),
                                       LogisticRegression(fit_intercept=False, max_iter=2000))
            self.model.fit(X, y, logisticregression__sample_weight=w)
        else:
            depth, lr, it, leaf = self.params
            self.model = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=it,
                                                        min_samples_leaf=leaf, l2_regularization=5.0,
                                                        random_state=0)
            self.model.fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]),
                           sample_weight=None if w is None else np.concatenate([w, w]))
        return self

    def predict(self, df):
        X = df[self.features].to_numpy(dtype=float) if hasattr(df, "columns") else np.asarray(df, dtype=float)
        return (self.model.predict_proba(X)[:, 1] + 1 - self.model.predict_proba(-X)[:, 1]) / 2


class CalibratedModel:
    """
    Corrige el exceso (o falta) de confianza: p' = sigmoide(a · logit(p)).
    a < 1 suaviza las probabilidades hacia 50 %. Sigue siendo simétrico (logit(1-p) = -logit(p)).
    'a' se ajusta en VALIDACIÓN con fit_scale, nunca en test.
    """

    def __init__(self, base, a=1.0):
        self.base, self.a = base, a
        self.name = f"{base.name} (calibrado, a={a:.3f})"
        self.features = base.features

    @staticmethod
    def apply(p, a):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-a * np.log(p / (1 - p))))

    @staticmethod
    def fit_scale(y, p, grid=np.arange(0.70, 1.301, 0.005)):
        return float(min(grid, key=lambda a: logloss(y, CalibratedModel.apply(p, a))))

    def fit(self, df):
        self.base.fit(df)
        return self

    def predict(self, df):
        return self.apply(self.base.predict(df), self.a)


class EnsembleModel:
    """Media de varios SymmetricModel (sigue siendo simétrica)."""

    def __init__(self, name, members, weights=None):
        self.name, self.members = name, members
        self.weights = weights or [1.0 / len(members)] * len(members)
        self.features = members[0].features

    def fit(self, df):
        for m in self.members:
            m.fit(df)
        return self

    def predict(self, df):
        return sum(w * m.predict(df) for w, m in zip(self.weights, self.members))
