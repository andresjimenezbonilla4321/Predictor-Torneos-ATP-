# Predictor de Torneos ATP

Modelo que estima la probabilidad de victoria en partidos ATP (circuito principal y
Challenger) y **simula torneos completos** por Monte Carlo para calcular la probabilidad
de que cada jugador llegue a cada ronda y gane el título. Incluye una **interfaz web**,
un **backtest con 678 torneos reales** y una **comparación con las casas de apuestas**.

## Interfaz web

```bash
streamlit run app.py
```

Se abre en el navegador (http://localhost:8501) con seis secciones:

| Sección | Qué hace |
|---|---|
| 🎾 Partido | Probabilidad de cada jugador, comparativa y **por qué** (qué factores pesan más) |
| 🏆 Torneo | Simula un cuadro: los N mejores por Elo, jugadores elegidos o un cuadro exacto; descarga en CSV |
| 📈 Ranking Elo | Ranking general y por superficie, tendencia de los últimos 90 días |
| 👤 Jugador | Ficha: Elo por superficie, evolución histórica, últimos partidos y balance |
| ✅ Fiabilidad | Calibración, acierto según la confianza y resultados del backtest de torneos |
| 💰 Casas de apuestas | Modelo frente a cuotas reales y apuestas simuladas |

En la barra lateral se elige la fecha (afecta a edad, ranking y días sin jugar).

## Resultados (test: 2024 – junio 2026, nunca vistos al entrenar)

| Modelo | Partidos | Acierto | Log-loss | Brier |
|---|---:|---:|---:|---:|
| Elo (fórmula, sin entrenar) | 23.134 | 63,2 % | 0,6432 | 0,2256 |
| Modelo anterior | 23.134 | 64,2 % | 0,6279 | 0,2194 |
| **Modelo final (calibrado)** – todos los partidos | 23.134 | **64,5 %** | **0,6268** | **0,2189** |
| Modelo final – sin retiradas | 22.303 | 65,0 % | 0,6224 | 0,2170 |

- Circuito principal: **66,6 %** · Challenger: 63,6 % · Grand Slams: **73,5 %**
- Cuando el modelo da ≥ 80 % al favorito, acierta un 86 % (8,5 % de los partidos).
- **Bien calibrado**: un 70 % predicho gana ~70 % de las veces (tabla en `train_best_report.txt`).

### Qué mejoró el modelo

| Cambio | Dónde se midió | Efecto |
|---|---|---|
| Elo con **margen de victoria** (un 6-1 6-1 cuenta más que un 7-6 en el tercero) y parámetros ajustados (`compute_elo.py --tune`, informe en `elo_tuning.txt`) | log-loss del Elo, 2005–2021 | 0,6196 → **0,6165** |
| Variables nuevas: **puntos ATP**, **tendencia del Elo** en 90 días, Elo × circuito principal | log-loss en validación 2022–2023 | 0,6276 → 0,6256 |
| Más peso a los partidos recientes al entrenar (regresión logística) | log-loss en validación | 0,6259 → 0,6249 |

Probado sin mejora: árboles más grandes y peso por recencia en los árboles.

## Comparación con las casas de apuestas

`compare_odds.py` cruza las predicciones de test con las cuotas reales del circuito principal
(conjunto público de Kaggle `dissfya/atp-tennis-2000-2023daily-pull`; no incluye Challengers).
La probabilidad de la casa se obtiene de sus cuotas quitando su margen (≈ 5 %). Informe en `odds_report.txt`.

| Mismos 5.881 partidos (2024 – junio 2026) | Acierto | Log-loss |
|---|---:|---:|
| Casa de apuestas | **69,0 %** | **0,5855** |
| Modelo | 66,9 % | 0,6011 |
| Elo | 65,5 % | 0,6164 |

- La casa es mejor de forma clara y significativa (diferencia de log-loss +0,016 [+0,011, +0,020]).
  Es lo esperable: las cuotas incorporan lesiones, noticias y el dinero de miles de apostantes.
- Modelo y casa eligen al mismo favorito en el 88 % de los partidos; cuando discrepan, acierta la casa
  el 59 % de las veces.
- Mezclar modelo y casa no mejora a la casa (el peso óptimo del modelo, elegido en validación, es 0).
- **Apuestas simuladas** apostando cuando el modelo ve "valor": se pierde dinero (ROI ≈ −8 % a −10 %).
  El modelo no sirve para ganar a las casas; sirve para estimar probabilidades razonables donde no hay
  cuotas (Challengers, cuadros futuros, simulaciones de torneo).

## Backtest de torneos (¿sirve para predecir torneos?)

`backtest_tournaments.py` simula los **678 torneos reales de 2024 – junio 2026** con lo que se
sabía antes de empezar cada uno (modelo entrenado solo con 2005–2023) y compara con lo que pasó.
Informe completo en `backtest_report.txt`.

| Grupo | Torneos | Log-loss del campeón: modelo | Solo Elo | Azar | Mejora frente al Elo (IC 95 %) |
|---|---:|---:|---:|---:|---|
| Todos | 678 | **2,802** | 2,915 | 3,502 | +0,114 [+0,067, +0,160] significativa |
| Grand Slam | 10 | **1,415** | 1,541 | 4,852 | +0,126 [−0,009, +0,261] no concluyente |
| Masters 1000 | 23 | **2,484** | 2,805 | 4,400 | +0,321 [+0,032, +0,610] significativa |
| ATP 250/500 | 118 | 2,582 | 2,562 | 3,398 | −0,020 [−0,115, +0,075] no concluyente |
| Challenger | 527 | **2,891** | 3,025 | 3,461 | +0,134 [+0,080, +0,189] significativa |

(El "solo Elo" ya es el Elo mejorado con margen de victoria; con el modelo anterior el log-loss global era 2,811.)

- **Calibrado también a nivel de torneo**: a los jugadores a los que se daba un 10–20 % de ser
  campeones, ganaron el 13,1 % de las veces (predicho 13,5 %); para llegar a semis, 25,9 % predicho
  frente a 25,9 % real.
- **Grandes favoritos**: con ≥ 35 % de ser campeón (39 casos) ganaron algo menos de lo previsto;
  con tan pocos casos la diferencia está dentro del margen de azar.
- **Grand Slams**: el campeón estuvo siempre entre los 4 favoritos del modelo y en 7 de 10 entre
  los 2 primeros (el duelo Sinner–Alcaraz hace que el favorito nº 1 solo ganara 2 de 10).
- Las cifras por grupo pequeño (Grand Slam, Masters) se mueven unas centésimas según la semilla de
  la simulación: con 10 o 23 torneos, el intervalo de confianza es lo que hay que mirar.

### ¿Hace falta una calibración distinta para torneos?

Probado y descartado (con el modelo anterior). En los torneos de **validación** (2022–2023, modelo entrenado solo con
2005–2021) se buscó el factor de calibración que mejor predice campeones (`backtest_tune_val.txt`):
el óptimo es a ≈ 0,80, prácticamente igual que el 0,875 que ya se usa para partidos (log-loss
2,7937 frente a 2,7938 con 0,85). En **test**, a = 0,80 frente a 0,875 da una mejora de +0,0001
[−0,007, +0,007]: ninguna. El simulador mantiene la calibración de partidos. Recalibrar de forma
global tampoco corrige la ligera sobreconfianza en los grandes favoritos, así que si existe es un
efecto específico de los jugadores de arriba, no de la calibración.

```bash
python backtest_tournaments.py                                  # todos los torneos de test
python backtest_tournaments.py --show "Roland Garros" --year 2025   # un torneo concreto
python backtest_tournaments.py --split val                      # torneos de validación (2022-2023)
python backtest_tournaments.py --tune-a                         # buscar la calibración de torneos en validación
python backtest_tournaments.py --compare-a 0.80                 # comparar otra calibración en test
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Uso

```bash
# 1) Descargar datos, calcular todo y entrenar (≈ 3 min; --backtest y --odds añaden las validaciones)
python run_pipeline.py

# 2) Abrir la interfaz web
streamlit run app.py

# 3) O desde la terminal: predecir un partido
python predict.py "Jannik Sinner" "Carlos Alcaraz" --surface Clay --best-of 5

# 4) Simular un torneo a partir de un cuadro
python simulate_tournament.py --draw draws/ejemplo_16_clay.txt --surface Clay

# ... o con los N mejores jugadores activos, colocados como cabezas de serie por Elo
python simulate_tournament.py --top 32 --surface Grass --best-of 5 --out wimbledon.csv
```

Formato del cuadro (`draws/*.txt`): un jugador por línea en el orden del cuadro (1 vs 2,
3 vs 4, …), `BYE` para huecos, `#` para comentarios. El tamaño debe ser potencia de 2,
o usa `--seed-by-elo` para que se ordene y complete solo. Los nombres se buscan sin
distinguir tildes ni mayúsculas, y vale el apellido (`"djokovic"`). Si el nombre es
ambiguo se elige al jugador con más partidos y se avisa: usa el nombre completo.

`--date AAAA-MM-DD` fija la fecha del partido/torneo (edad, ranking, días sin jugar).
Por defecto se usa una semana después del último torneo de los datos.

## Pipeline

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `data_set.py` | Descarga partidos, jugadores y rankings 2000–2026 |
| 2 | `clean_matches.py` | Limpieza documentada en `cleaning_report_matches.txt` |
| 3 | `compute_elo.py` | Elo general y por superficie: K variable, margen de victoria, peso del mejor de 5, inactividad y tendencia (`--tune` ajusta los parámetros en 2005–2021) |
| 4 | `add_features.py` | Forma, descanso, inactividad, cansancio y cara a cara |
| 5 | `add_serve_stats.py` | % puntos ganados al saque/resto, aces, dobles faltas, 1er saque (suavizados) |
| 6 | `build_model_dataset.py` | Jugadores A/B al azar, diferencias A − B, split temporal |
| 7 | `train_best.py` | Elige modelo por log-loss en validación, calibra, evalúa en test; guarda `models/best_model.joblib` (todos los datos) y `models/model_trainval.joblib` (2005–2023) y `models/model_train.joblib` (2005–2021) para el backtest |
| — | `train_baseline.py` | Comparativa: cuánto aporta cada grupo de características |
| — | `predict.py` | Predicción de partidos futuros con el estado actual de cada jugador |
| — | `simulate_tournament.py` | Simulación Monte Carlo de un cuadro |
| — | `backtest_tournaments.py` | Simula los torneos reales de test y mide si las predicciones de torneo aciertan |
| — | `compare_odds.py` | Compara el modelo con las cuotas de las casas de apuestas (descarga las cuotas en `odds/`) |
| — | `app.py` | Interfaz web (Streamlit) |

`common.py` contiene la configuración y las funciones compartidas: el entrenamiento y la
predicción calculan las características con **el mismo código**.

## Decisiones de metodología

- **Sin fuga de datos**: en cada partido se guarda lo que se sabía *antes* de jugarlo y
  solo después se actualiza el historial. Las medias del circuito para el suavizado del
  saque se calculan solo con los años de entrenamiento.
- **División temporal**: train 2005–2021 · validación 2022–2023 · test 2024+. El test se mira una vez.
- **Modelo simétrico**: P(A gana) = 1 − P(B gana) siempre (regresión sin intercepto y
  árboles entrenados con partidos en espejo).
- **Selección por log-loss**, no por acierto: el acierto es más ruidoso y no premia
  probabilidades correctas.
- **Calibración** `p' = σ(a·logit p)` ajustada en validación (a ≈ 0,93: el modelo sin
  calibrar era algo sobreconfiado).
- **Retiradas**: se quitan del entrenamiento pero se mantienen en validación y test,
  porque antes del partido no se sabe si habrá retirada.
- **Elo con inactividad**: tras > 120 días sin jugar, el Elo se acerca a 1500 (hasta un
  30 % de la distancia por año fuera).
- **Todos los ajustes** (parámetros del Elo, variables, pesos) se eligen con train o validación;
  test solo se usa para el informe final.

## Limitaciones conocidas

- Todos los partidos de un torneo comparten la fecha de inicio: el descanso se mide entre torneos.
- En la simulación se usa el estado de antes del torneo en todas las rondas (no se modela
  el cansancio acumulado dentro del torneo).
- Los datos acaban el 2026-06-01; para predecir más adelante hay que volver a ejecutar el pipeline.

## Tests

```bash
python -m pytest -q
```

## Datos

Datos de partidos, jugadores y rankings de **Jeff Sackmann**
([tennis_atp](https://github.com/JeffSackmann/tennis_atp)), licencia
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), descargados del
espejo actualizado `Aneeshers/tennis-sackmann-archive`. Uso no comercial.

Cuotas: conjunto público de Kaggle
[dissfya/atp-tennis-2000-2023daily-pull](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)
(resultados y cuotas del circuito principal ATP).

Las probabilidades del modelo son estimaciones estadísticas, no consejos de apuesta.
