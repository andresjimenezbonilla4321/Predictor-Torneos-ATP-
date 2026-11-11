# Predictor de Torneos ATP

Modelo que estima la probabilidad de victoria en partidos ATP (circuito principal y
Challenger) y **simula torneos completos** por Monte Carlo para calcular la probabilidad
de que cada jugador llegue a cada ronda y gane el título.

## Resultados (test: 2024 – junio 2026, nunca vistos al entrenar)

| Modelo | Partidos | Acierto | Log-loss | Brier |
|---|---:|---:|---:|---:|
| Elo (fórmula, sin entrenar) | 23.134 | 63,0 % | 0,6448 | 0,2264 |
| **Modelo final (calibrado)** – todos los partidos | 23.134 | **64,2 %** | **0,6279** | **0,2194** |
| Modelo final – sin retiradas | 22.303 | 64,7 % | 0,6232 | 0,2173 |

- Circuito principal: **66,4 %** · Challenger: 63,2 % · Grand Slams: **73,6 %**
- Cuando el modelo da ≥ 80 % al favorito, acierta un 83 % (12 % de los partidos).
- **Bien calibrado**: un 70 % predicho gana ~70 % de las veces (tabla en `train_best_report.txt`),
  requisito para que las simulaciones de torneos tengan sentido.

## Backtest de torneos (¿sirve para predecir torneos?)

`backtest_tournaments.py` simula los **678 torneos reales de 2024 – junio 2026** con lo que se
sabía antes de empezar cada uno (modelo entrenado solo con 2005–2023) y compara con lo que pasó.
Informe completo en `backtest_report.txt`.

| Grupo | Torneos | Log-loss del campeón: modelo | Solo Elo | Azar | Mejora frente al Elo (IC 95 %) |
|---|---:|---:|---:|---:|---|
| Todos | 678 | **2,811** | 2,915 | 3,502 | +0,105 [+0,063, +0,147] significativa |
| Grand Slam | 10 | **1,415** | 1,580 | 4,852 | +0,165 [+0,003, +0,327] significativa |
| Masters 1000 | 23 | **2,594** | 2,805 | 4,400 | +0,211 [−0,031, +0,452] no concluyente |
| ATP 250/500 | 118 | 2,571 | 2,550 | 3,398 | −0,020 [−0,105, +0,064] no concluyente |
| Challenger | 527 | **2,900** | 3,027 | 3,461 | +0,127 [+0,078, +0,176] significativa |

- **Calibrado también a nivel de torneo**: a los jugadores a los que se daba un 10–20 % de ser
  campeones, ganaron el 13,1 % de las veces (predicho 13,4 %); para llegar a semis, 25,9 % predicho
  frente a 26,3 % real.
- **Grandes favoritos**: con ≥ 35 % de ser campeón ganaron algo menos de lo previsto (44 casos:
  43,4 % predicho, 38,6 % real; en validación 2022–2023, ~42 % frente a ~32 %). La diferencia está
  dentro del margen de azar con tan pocos casos: hay indicios, no una prueba.
- **Grand Slams**: el campeón estuvo siempre entre los 4 favoritos del modelo y en 8 de 10 entre
  los 2 primeros (el duelo Sinner–Alcaraz hace que el favorito nº 1 solo ganara 2 de 10).
- Las cifras por grupo pequeño (Grand Slam, Masters) se mueven unas centésimas según la semilla de
  la simulación: con 10 o 23 torneos, el intervalo de confianza es lo que hay que mirar.

### ¿Hace falta una calibración distinta para torneos?

Probado y descartado. En los torneos de **validación** (2022–2023, modelo entrenado solo con
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
# 1) Descargar datos, calcular todo y entrenar (≈ 3 min; añade --backtest para validar torneos)
python run_pipeline.py

# 2) Predecir un partido
python predict.py "Jannik Sinner" "Carlos Alcaraz" --surface Clay --best-of 5

# 3) Simular un torneo a partir de un cuadro
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
| 3 | `compute_elo.py` | Elo general y por superficie, K variable, ajuste por inactividad (`--tune` para elegir parámetros) |
| 4 | `add_features.py` | Forma, descanso, inactividad, cansancio y cara a cara |
| 5 | `add_serve_stats.py` | % puntos ganados al saque/resto, aces, dobles faltas, 1er saque (suavizados) |
| 6 | `build_model_dataset.py` | Jugadores A/B al azar, diferencias A − B, split temporal |
| 7 | `train_best.py` | Elige modelo por log-loss en validación, calibra, evalúa en test; guarda `models/best_model.joblib` (todos los datos) y `models/model_trainval.joblib` (2005–2023) y `models/model_train.joblib` (2005–2021) para el backtest |
| — | `train_baseline.py` | Comparativa: cuánto aporta cada grupo de características |
| — | `predict.py` | Predicción de partidos futuros con el estado actual de cada jugador |
| — | `simulate_tournament.py` | Simulación Monte Carlo de un cuadro |
| — | `backtest_tournaments.py` | Simula los torneos reales de test y mide si las predicciones de torneo aciertan |

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
- **Calibración** `p' = σ(a·logit p)` ajustada en validación (a ≈ 0,875: el modelo sin
  calibrar era algo sobreconfiado).
- **Retiradas**: se quitan del entrenamiento pero se mantienen en validación y test,
  porque antes del partido no se sabe si habrá retirada.
- **Elo con inactividad**: tras > 120 días sin jugar, el Elo se acerca a 1500 (hasta un
  20 % de la distancia por año fuera). Mejora el log-loss del Elo de 0,6257 a 0,6238 (2005–2021).

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
