"""
Interfaz web del Predictor ATP.

Ejecutar:   streamlit run app.py
(antes hay que haber ejecutado el pipeline: python run_pipeline.py)
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import FEATURES, CalibratedModel, expected
from predict import Predictor
from simulate_tournament import BYE, read_draw, simulate_draw, top_players

ROOT = Path(__file__).resolve().parent
SURFACES = {"Dura": "Hard", "Tierra": "Clay", "Hierba": "Grass"}
SURF_ES = {v: k for k, v in SURFACES.items()} | {"Carpet": "Moqueta"}
FEATURE_NAMES = {
    "elo_diff": "Elo general", "elo_surf_diff": "Elo en la superficie", "elo_diff_bo5": "Elo × mejor de 5",
    "elo_diff_main": "Elo × circuito principal", "elo_trend_diff": "Tendencia del Elo (90 días)",
    "log_rank_diff": "Ranking ATP", "log_points_diff": "Puntos ATP", "age_diff": "Edad", "ht_diff": "Altura",
    "lefty_diff": "Zurdo", "exp_diff": "Experiencia", "form_diff": "Forma reciente", "rest_diff": "Descanso",
    "layoff_diff": "Días sin jugar", "games14_diff": "Juegos en 14 días (cansancio)", "h2h_diff": "Cara a cara",
    "spw_diff": "Puntos ganados al saque", "rpw_diff": "Puntos ganados al resto", "ace_diff": "Aces",
    "df_diff": "Dobles faltas", "fin_diff": "Primeros saques dentro",
}

st.set_page_config(page_title="Predictor ATP", page_icon="🎾", layout="wide")


# ------------------------------------------------------------------ datos
@st.cache_resource(show_spinner="Cargando modelo y estado de los jugadores…")
def get_predictor(date):
    return Predictor(date=date)


@st.cache_data(show_spinner=False)
def load_csv(name, **kw):
    path = ROOT / name
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data(show_spinner="Leyendo el historial de partidos…")
def load_matches():
    cols = ["tourney_date", "tourney_name", "tourney_level", "surface", "round", "winner_id", "loser_id",
            "winner_name", "loser_name", "score", "elo_w", "elo_l", "elo_surf_w", "elo_surf_l"]
    return pd.read_csv(ROOT / "atp_matches_elo.csv", usecols=cols, parse_dates=["tourney_date"])


@st.cache_data(show_spinner=False)
def player_options(_pr, date_key, include_inactive):
    """(id, etiqueta) de los jugadores, ordenados por Elo actual."""
    last = _pr.elo["last_day"]
    ids = [p for p in _pr.elo["gen_r"] if include_inactive or _pr.day - last.get(p, -10**9) <= 365]
    ids.sort(key=lambda p: -_pr.current_elo(p))
    return [(p, f"{_pr.name(p)} · Elo {_pr.current_elo(p):.0f}") for p in ids]


def pct(x):
    return f"{x * 100:.1f} %"


# ------------------------------------------------------------------ barra lateral
pr0 = get_predictor(None)
with st.sidebar:
    st.title("🎾 Predictor ATP")
    default_date = pr0.date.date()
    date = st.date_input("Fecha del partido / torneo", value=default_date, min_value=pr0.data_end.date(),
                         help="Afecta a la edad, el ranking, los días sin jugar y el ajuste por inactividad del Elo.")
    pr = get_predictor(str(date))
    st.caption(f"Datos hasta **{pr.data_end.date()}** · ranking ATP del **{pr.ranking_date}**")
    st.caption(f"Modelo: {pr.model_name}")
    include_inactive = st.toggle("Incluir jugadores inactivos", value=False,
                                 help="Por defecto solo aparecen quienes han jugado en el último año.")
    st.divider()
    st.caption("Datos de partidos: Jeff Sackmann (CC BY-NC-SA 4.0). Cuotas: conjunto público de Kaggle. "
               "Uso no comercial. Las probabilidades son estimaciones, no consejos de apuesta.")

options = player_options(pr, str(date), include_inactive)
labels = {p: lab for p, lab in options}
ids = [p for p, _ in options]

tabs = st.tabs(["🎾 Partido", "🏆 Torneo", "📈 Ranking Elo", "👤 Jugador", "✅ Fiabilidad del modelo",
                "💰 Casas de apuestas"])

# ================================================================== PARTIDO
with tabs[0]:
    st.subheader("¿Quién gana?")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Jugador A", ids, index=0, format_func=labels.get, key="pa")
    b = c2.selectbox("Jugador B", ids, index=1 if len(ids) > 1 else 0, format_func=labels.get, key="pb")
    c3, c4, c5 = st.columns(3)
    surface = SURFACES[c3.radio("Superficie", list(SURFACES), horizontal=True, key="m_surf")]
    best_of = c4.radio("Sets", [3, 5], horizontal=True, format_func=lambda x: f"Al mejor de {x}", key="m_bo")
    is_main = c5.radio("Circuito", ["Principal (ATP)", "Challenger"], horizontal=True, key="m_lvl") == "Principal (ATP)"

    if a == b:
        st.warning("Elige dos jugadores distintos.")
    else:
        pa_, pb_ = pr.profile(a, surface), pr.profile(b, surface)
        p = float(pr.pairs_proba([(a, b)], surface, best_of, is_main=int(is_main))[0])
        na, nb = pr.name(a), pr.name(b)

        m1, m2, m3 = st.columns([2, 2, 1])
        m1.metric(na, pct(p))
        m2.metric(nb, pct(1 - p))
        m3.metric("Solo Elo", pct(expected(pa_["elo"], pb_["elo"])), help=f"Probabilidad de {na} según la fórmula del Elo")
        bar = pd.DataFrame({"Jugador": [na, nb], "Probabilidad": [p, 1 - p], "orden": [0, 1]})
        st.altair_chart(
            alt.Chart(bar).mark_bar(height=34).encode(
                x=alt.X("sum(Probabilidad):Q", stack="normalize", axis=alt.Axis(format="%", title=None)),
                color=alt.Color("Jugador:N", sort=[na, nb], legend=alt.Legend(orient="bottom", title=None)),
                order="orden:Q", tooltip=["Jugador", alt.Tooltip("Probabilidad:Q", format=".1%")],
            ).properties(height=70), width="stretch")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Comparativa**")
            h2h = pr.hist["h2h"]
            rows = [("Elo general", "elo", "{:.0f}"), (f"Elo en {SURF_ES[surface].lower()}", "elo_surf", "{:.0f}"),
                    ("Tendencia Elo (90 días)", "trend", "{:+.0f}"), ("Ranking ATP", "rank", "{:.0f}"),
                    ("Puntos ATP", "points", "{:.0f}"), ("Edad", "age", "{:.1f}"),
                    ("Forma (últimos 10, suavizada)", "form", "{:.2f}"), ("Días sin jugar", "layoff", "{:.0f}"),
                    ("% puntos ganados al saque", "spw", "{:.1%}"), ("% puntos ganados al resto", "rpw", "{:.1%}"),
                    ("Partidos en los datos", "n", "{:.0f}")]
            comp = pd.DataFrame(
                [(lab, fmt.format(pa_[k]) if pd.notna(pa_[k]) else "–", fmt.format(pb_[k]) if pd.notna(pb_[k]) else "–")
                 for lab, k, fmt in rows] + [("Cara a cara", str(h2h.get((a, b), 0)), str(h2h.get((b, a), 0)))],
                columns=["", na, nb]).set_index("")
            st.dataframe(comp, width="stretch")
        with right:
            st.markdown("**¿Por qué?** Lo que más inclina la balanza")
            model = pr.model
            base = model.base if isinstance(model, CalibratedModel) else model
            if getattr(base, "kind", None) == "logreg":
                A = {k: np.array([pa_[k]], dtype=object if k == "hand" else float) for k in pa_ if k != "id"}
                B = {k: np.array([pb_[k]], dtype=object if k == "hand" else float) for k in pb_ if k != "id"}
                A["h2h"], B["h2h"] = np.array([h2h.get((a, b), 0.0)]), np.array([h2h.get((b, a), 0.0)])
                from common import make_features
                x = pd.DataFrame(make_features(A, B, [best_of], [int(is_main)]))[FEATURES].iloc[0].to_numpy()
                pipe = base.model
                contrib = pipe[-1].coef_[0] * x / pipe[0].scale_
                cdf = pd.DataFrame({"Factor": [FEATURE_NAMES.get(f, f) for f in FEATURES], "Peso": contrib})
                cdf = cdf.reindex(cdf["Peso"].abs().sort_values(ascending=False).index).head(8)
                cdf["A favor de"] = np.where(cdf["Peso"] > 0, na, nb)
                st.altair_chart(
                    alt.Chart(cdf).mark_bar().encode(
                        x=alt.X("Peso:Q", title=f"← {nb}   ·   {na} →"),
                        y=alt.Y("Factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                        color=alt.Color("A favor de:N", sort=[na, nb], legend=None),
                        tooltip=["Factor", "A favor de", alt.Tooltip("Peso:Q", format=".2f")],
                    ).properties(height=260), width="stretch")
                st.caption("Contribución de cada factor al resultado del modelo (escala logit).")
            else:
                st.info("La explicación por factores solo está disponible con el modelo de regresión logística.")

# ================================================================== TORNEO
with tabs[1]:
    st.subheader("Simular un torneo")
    mode = st.radio("¿Cómo se forma el cuadro?",
                    ["Los mejores por Elo", "Elegir jugadores", "Cuadro exacto (texto)"], horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    t_surface = SURFACES[c1.selectbox("Superficie", list(SURFACES), key="t_surf")]
    t_bo = c2.selectbox("Sets", [3, 5], format_func=lambda x: f"Al mejor de {x}", key="t_bo")
    t_main = c3.selectbox("Circuito", ["Principal (ATP)", "Challenger"], key="t_lvl") == "Principal (ATP)"
    sims = c4.selectbox("Simulaciones", [5000, 20000, 50000], index=1)

    entries, seed_by_elo, error = None, True, None
    if mode == "Los mejores por Elo":
        n_top = st.slider("Número de jugadores (se colocan como cabezas de serie por Elo; huecos = BYE)", 4, 128, 32)
        entries = top_players(pr, n_top)
    elif mode == "Elegir jugadores":
        chosen = st.multiselect("Jugadores (se colocan como cabezas de serie por Elo)", ids,
                                default=ids[:8], format_func=labels.get)
        entries = chosen
    else:
        seed_by_elo = False
        example = ROOT / "draws" / "ejemplo_16_clay.txt"
        default_txt = "\n".join(read_draw(example)) if example.exists() else ""
        txt = st.text_area("Un jugador por línea, en orden de cuadro (1 contra 2, 3 contra 4…). Usa BYE para huecos. "
                           "El total debe ser potencia de 2.", value=default_txt, height=260)
        try:
            entries = [BYE if ln.strip().upper() == "BYE" else pr.find_player(ln.strip())
                       for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        except ValueError as e:
            error = str(e)

    if error:
        st.error(error)
    elif st.button("▶ Simular torneo", type="primary"):
        try:
            with st.spinner("Jugando el torneo miles de veces…"):
                res = simulate_draw(pr, entries, t_surface, t_bo, sims, seed_by_elo, int(t_main))
            st.session_state["sim"] = res
        except ValueError as e:
            st.error(str(e))

    res = st.session_state.get("sim")
    if res:
        table = res["table"]
        rounds = [c for c in table.columns if c not in ("Jugador", "Elo") and not c.startswith("Elo ")]
        st.markdown(f"**{res['n']} jugadores · cuadro de {res['size']} · {res['sims']:,} simulaciones**")
        top = table.head(16)
        st.altair_chart(
            alt.Chart(top).mark_bar().encode(
                x=alt.X("Campeón:Q", title="Probabilidad de ser campeón (%)"),
                y=alt.Y("Jugador:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                tooltip=["Jugador", alt.Tooltip("Campeón:Q", format=".1f")],
            ).properties(height=max(200, 28 * len(top))), width="stretch")
        st.dataframe(
            table, width="stretch", height=min(600, 38 + 35 * len(table)),
            column_config={r: st.column_config.ProgressColumn(r, format="%.1f %%", min_value=0, max_value=100)
                           for r in rounds})
        st.download_button("Descargar CSV", table.to_csv(index_label="puesto").encode("utf-8"),
                           "simulacion_torneo.csv", "text/csv")
        with st.expander("Primera ronda"):
            fr = pd.DataFrame([(x, y, "" if p is None else f"{p * 100:.0f} % – {(1 - p) * 100:.0f} %")
                               for x, y, p in res["first_round"]], columns=["Jugador", "Rival", "Probabilidades"])
            st.dataframe(fr, width="stretch", hide_index=True)

# ================================================================== RANKING ELO
with tabs[2]:
    st.subheader("Ranking Elo")
    c1, c2 = st.columns([1, 2])
    sort_by = c1.selectbox("Ordenar por", ["Elo general", "Elo dura", "Elo tierra", "Elo hierba", "Tendencia"])
    search = c2.text_input("Buscar jugador", "")
    rows = []
    for p in ids:
        e = pr.elo
        rows.append({
            "Jugador": pr.name(p), "Elo general": round(pr.current_elo(p)),
            "Elo dura": round(e["srf_r"][(p, "Hard")]) if (p, "Hard") in e["srf_r"] else None,
            "Elo tierra": round(e["srf_r"][(p, "Clay")]) if (p, "Clay") in e["srf_r"] else None,
            "Elo hierba": round(e["srf_r"][(p, "Grass")]) if (p, "Grass") in e["srf_r"] else None,
            "Ranking ATP": pr.rank.get(p), "Partidos": e["gen_n"].get(p, 0),
        })
    rk = pd.DataFrame(rows)
    if sort_by == "Tendencia":
        from compute_elo import elo_trend
        rk["Tendencia"] = [round(elo_trend(pr.elo.get("hist_d", {}), pr.elo.get("hist_r", {}), p,
                                           pr.current_elo(p), pr.day)) for p in ids]
    rk = rk.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)
    rk.index += 1
    if search:
        rk = rk[rk["Jugador"].str.contains(search, case=False)]
    top20 = rk.dropna(subset=[sort_by]).head(20)
    if len(top20):
        lo, hi = float(top20[sort_by].min()), float(top20[sort_by].max())
        pad = max(10.0, (hi - lo) * 0.1)
        st.altair_chart(
            alt.Chart(top20).mark_bar(clip=True).encode(
                x=alt.X(f"{sort_by}:Q", scale=alt.Scale(domain=[lo - pad, hi + pad / 2])),
                y=alt.Y("Jugador:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                tooltip=list(top20.columns)[:6],
            ).properties(height=28 * len(top20)), width="stretch")
    st.dataframe(rk, width="stretch", height=500)

# ================================================================== JUGADOR
with tabs[3]:
    st.subheader("Ficha de jugador")
    pid = st.selectbox("Jugador", ids, format_func=labels.get, key="pj")
    prof = {s: pr.profile(pid, s) for s in ("Hard", "Clay", "Grass")}
    k = st.columns(6)
    k[0].metric("Elo general", f"{prof['Hard']['elo']:.0f}", f"{prof['Hard']['trend']:+.0f} en 90 días")
    k[1].metric("Elo dura", f"{prof['Hard']['elo_surf']:.0f}")
    k[2].metric("Elo tierra", f"{prof['Clay']['elo_surf']:.0f}")
    k[3].metric("Elo hierba", f"{prof['Grass']['elo_surf']:.0f}")
    rank = prof["Hard"]["rank"]
    k[4].metric("Ranking ATP", "–" if pd.isna(rank) else f"{rank:.0f}")
    k[5].metric("Forma", f"{prof['Hard']['form']:.2f}", help="Victorias en los últimos 10 partidos, suavizado hacia 0,5")

    mt = load_matches()
    pm = mt[(mt["winner_id"] == pid) | (mt["loser_id"] == pid)].copy()
    if len(pm):
        won = pm["winner_id"] == pid
        pm["Elo"] = np.where(won, pm["elo_w"], pm["elo_l"])
        pm["Rival"] = np.where(won, pm["loser_name"], pm["winner_name"])
        pm["Resultado"] = np.where(won, "Victoria", "Derrota")
        hist = pm.groupby("tourney_date", as_index=False)["Elo"].first()
        st.altair_chart(
            alt.Chart(hist).mark_line().encode(
                x=alt.X("tourney_date:T", title=None), y=alt.Y("Elo:Q", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("tourney_date:T", title="Fecha"), alt.Tooltip("Elo:Q", format=".0f")],
            ).properties(height=280, title="Evolución del Elo general (antes de cada torneo)"),
            width="stretch")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("**Últimos partidos**")
            last = pm.sort_values(["tourney_date"], ascending=False).head(20)
            last = last.assign(Fecha=last["tourney_date"].dt.date, Superficie=last["surface"].map(SURF_ES))
            st.dataframe(last[["Fecha", "tourney_name", "round", "Superficie", "Rival", "Resultado", "score"]]
                         .rename(columns={"tourney_name": "Torneo", "round": "Ronda", "score": "Marcador"}),
                         width="stretch", hide_index=True)
        with c2:
            st.markdown("**Balance último año por superficie**")
            recent = pm[pm["tourney_date"] >= pm["tourney_date"].max() - pd.Timedelta(days=365)]
            bal = recent.groupby("surface")["Resultado"].value_counts().unstack(fill_value=0)
            bal.index = bal.index.map(lambda s: SURF_ES.get(s, s))
            st.dataframe(bal, width="stretch")

# ================================================================== FIABILIDAD
with tabs[4]:
    st.subheader("¿Me puedo fiar del modelo?")
    st.caption("Todo se mide con partidos y torneos de 2024 en adelante que el modelo NO vio al entrenar.")
    pb = load_csv("predictions_best.csv")
    if pb is not None:
        main_ = pb["level_source"] == "main"
        k = st.columns(4)
        k[0].metric("Acierto (todos)", pct(pb["hit"].mean()), help=f"{len(pb):,} partidos de test")
        k[1].metric("Circuito principal", pct(pb.loc[main_, "hit"].mean()))
        k[2].metric("Challenger", pct(pb.loc[~main_, "hit"].mean()))
        conf80 = pb["confidence"] >= 0.8
        k[3].metric("Cuando da ≥ 80 %", pct(pb.loc[conf80, "hit"].mean()),
                    help=f"{conf80.mean() * 100:.0f} % de los partidos")
        bins = np.arange(0.5, 1.0001, 0.05)
        pb["tramo"] = pd.cut(pb["confidence"], bins, include_lowest=True)
        cal = pb.groupby("tramo", observed=True).agg(predicho=("confidence", "mean"), real=("hit", "mean"),
                                                     partidos=("hit", "size")).reset_index(drop=True)
        diag = pd.DataFrame({"predicho": [0.5, 1.0], "real": [0.5, 1.0]})
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                (alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="predicho:Q", y="real:Q")
                 + alt.Chart(cal).mark_line(point=True).encode(
                     x=alt.X("predicho:Q", title="Probabilidad del favorito según el modelo", axis=alt.Axis(format="%"),
                             scale=alt.Scale(domain=[0.5, 1])),
                     y=alt.Y("real:Q", title="Ganó de verdad", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.5, 1])),
                     tooltip=[alt.Tooltip("predicho:Q", format=".1%"), alt.Tooltip("real:Q", format=".1%"), "partidos"]))
                .properties(height=320, title="Calibración: los puntos deben caer sobre la diagonal"),
                width="stretch")
        with c2:
            ths = np.arange(0.5, 0.91, 0.05)
            acc = pd.DataFrame({"umbral": ths,
                                "acierto": [pb.loc[pb["confidence"] >= t, "hit"].mean() for t in ths],
                                "partidos": [(pb["confidence"] >= t).mean() for t in ths]})
            st.altair_chart(
                alt.Chart(acc).mark_line(point=True).encode(
                    x=alt.X("umbral:Q", title="Confianza mínima del modelo", axis=alt.Axis(format="%")),
                    y=alt.Y("acierto:Q", title="Acierto", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("umbral:Q", format=".0%"), alt.Tooltip("acierto:Q", format=".1%"),
                             alt.Tooltip("partidos:Q", format=".0%", title="% de partidos")])
                .properties(height=320, title="Cuanto más seguro está, más acierta"), width="stretch")

    bt = load_csv("backtest_tournaments.csv")
    if bt is not None:
        st.markdown("#### Torneos reales simulados antes de jugarse")
        eps = 1e-4
        g = bt.groupby("level").apply(lambda d: pd.Series({
            "Torneos": len(d),
            "Log-loss modelo": -np.log(d["p_champ_model"].clip(lower=eps)).mean(),
            "Log-loss solo Elo": -np.log(d["p_champ_elo"].clip(lower=eps)).mean(),
            "Campeón entre los 4 favoritos": (d["champ_rank_model"] <= 4).mean(),
        }), include_groups=False)
        st.dataframe(g.style.format({"Torneos": "{:.0f}", "Log-loss modelo": "{:.3f}", "Log-loss solo Elo": "{:.3f}",
                                     "Campeón entre los 4 favoritos": "{:.0%}"}), width="stretch")
        st.caption("Log-loss del campeón: cuanto más bajo, más probabilidad le daba el modelo al que acabó ganando.")
        players = load_csv("atp_players.csv", low_memory=False)
        names = dict(zip(players["player_id"], players["name_first"] + " " + players["name_last"]))
        gs = bt[bt["level"] == "Grand Slam"].sort_values("date")
        if len(gs):
            st.markdown("**Grand Slams**")
            st.dataframe(pd.DataFrame({
                "Fecha": gs["date"], "Torneo": gs["tourney_name"], "Campeón": gs["champion"].map(names),
                "Prob. que le daba el modelo": (gs["p_champ_model"] * 100).round(1).astype(str) + " %",
                "Era el favorito nº": gs["champ_rank_model"], "Favorito del modelo": gs["favorite"].map(names),
            }), width="stretch", hide_index=True)

# ================================================================== CASAS DE APUESTAS
with tabs[5]:
    st.subheader("Modelo frente a casas de apuestas")
    om = load_csv("odds_matches.csv")
    if om is None:
        st.info("Ejecuta `python compare_odds.py` para generar la comparación.")
    else:
        y = om["target"].to_numpy()

        def stats(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return ((p > 0.5) == (y == 1)).mean(), -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

        st.caption(f"{len(om):,} partidos del circuito principal (2024 en adelante) con cuotas. "
                   "La probabilidad de la casa se calcula a partir de sus cuotas quitando su margen.")
        k = st.columns(3)
        for col, (lab, colname) in zip(k, [("Casa de apuestas", "p_book"), ("Modelo", "p_model"),
                                           ("Elo", "elo_prob_A")]):
            acc_, ll_ = stats(om[colname].to_numpy())
            col.metric(lab, pct(acc_), f"log-loss {ll_:.4f}", delta_color="off")
        same = (om["p_model"] > 0.5) == (om["p_book"] > 0.5)
        st.markdown(f"El modelo y la casa eligen al mismo favorito en el **{same.mean() * 100:.1f} %** de los partidos. "
                    f"Cuando no coinciden ({(~same).sum():,} partidos), acierta el modelo el "
                    f"**{((om.loc[~same, 'p_model'] > 0.5) == (om.loc[~same, 'target'] == 1)).mean() * 100:.1f} %**.")
        by = om.groupby("Series").apply(lambda d: pd.Series({
            "Partidos": len(d),
            "Acierto modelo": ((d["p_model"] > 0.5) == (d["target"] == 1)).mean(),
            "Acierto casa": ((d["p_book"] > 0.5) == (d["target"] == 1)).mean()}), include_groups=False)
        st.dataframe(by.style.format({"Partidos": "{:.0f}", "Acierto modelo": "{:.1%}", "Acierto casa": "{:.1%}"}),
                     width="stretch")

        st.markdown("#### Apuestas simuladas")
        st.caption("Ejercicio: apostar 1 unidad cuando el modelo ve más probabilidad que la que implica la cuota. "
                   "No es una recomendación de apuesta.")
        edge = st.slider("Ventaja mínima exigida (probabilidad del modelo × cuota − 1)", 0.0, 0.3, 0.05, 0.01)
        pA, oA, oB = om["p_model"].to_numpy(), om["odd_A"].to_numpy(), om["odd_B"].to_numpy()
        betA = pA * oA > 1 + edge
        betB = ((1 - pA) * oB > 1 + edge) & ~betA
        profit = np.where(betA, np.where(y == 1, oA - 1, -1), 0) + np.where(betB, np.where(y == 0, oB - 1, -1), 0)
        bets = betA | betB
        k = st.columns(3)
        k[0].metric("Apuestas", f"{bets.sum():,}")
        k[1].metric("Beneficio", f"{profit.sum():+.1f} u")
        k[2].metric("Rentabilidad (ROI)", f"{profit.sum() / max(bets.sum(), 1) * 100:+.1f} %")
        curve = pd.DataFrame({"Apuesta nº": np.arange(1, bets.sum() + 1), "Beneficio acumulado": profit[bets].cumsum()})
        if len(curve):
            st.altair_chart(alt.Chart(curve).mark_line().encode(x="Apuesta nº:Q", y="Beneficio acumulado:Q")
                            .properties(height=260), width="stretch")
