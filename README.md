# Data Mining / PSet #5 – NYC Taxi Trips: Ensemble ML Pipeline para Predicción de total_amount

## Equipo
- **Ahmed Puco** (00320082)
- **Martina Vásconez** (00324077)
- **Ma. Eulalia Moncayo** (00326226)
- **Joel Cuascota** (00327494)

**Boosting asignado:** XGBoost

---

## Descripción

Este proyecto implementa un **pipeline de Machine Learning de punta a punta** para predecir `total_amount` (monto total del viaje) usando datos históricos de taxis de NYC. Se comparan múltiples técnicas de ensemble learning:

- **Voting Regressor** (3 modelos base)
- **Bagging vs Pasting** (con árboles de decisión)
- **Boosting:** AdaBoost, Gradient Boosting, **XGBoost** (estudio profundo), LightGBM, CatBoost

El proyecto utiliza:
- **Docker Compose** con Jupyter + Spark
- **Snowflake** como data warehouse (esquemas `RAW` y `ANALYTICS`)
- **One Big Table (OBT)** desnormalizada para análisis directo
- **TimeSeriesSplit** para validación temporal
- **GridSearch/RandomSearch** para tuning de hiperparámetros
- **SHAP** para interpretabilidad del modelo ganador

---

## Problema de negocio

**Meta:** Estimar el `total_amount` de un viaje **al momento del pickup** para:
- **Pricing dinámico** que ajuste tarifas según demanda y condiciones
- **Gestión de promociones** basadas en predicciones de ingresos
- **Planificación de demanda** y optimización de flota

**Target:** `total_amount` (USD)  
**Métrica principal:** RMSE (Root Mean Squared Error)

---

## Decisiones de alcance y limitaciones de recursos

### Problema principal: Volumen de datos vs. Recursos computacionales

El dataset completo de NYC Taxi (2015-2025) contiene más de **850M de registros**, lo que representa:
- ~80GB en archivos Parquet comprimidos
- ~300GB+ en Snowflake con índices y OBT completa
- Tiempos de procesamiento de 8-12 horas para ingesta completa

### Solución implementada: Scope reducido estratégico

Para garantizar la **entrega funcional** del proyecto dentro de las restricciones de recursos, se tomaron las siguientes decisiones:

#### 1. Ingesta RAW (4 años: 2015-2018)
 **Se ingestan 4 años completos** en las tablas RAW:
- `RAW.YELLOW_TAXIS`: ~490M registros (2015-2018)
- `RAW.GREEN_TAXIS`: ~56M registros (2015-2018)
- Total: **~364M registros**

**Justificación:** 
- Periodo suficientemente largo para capturar estacionalidad y tendencias
- Balance entre volumen de datos y recursos computacionales disponibles
- Permite validación temporal robusta (3 años train + 1 año val/test)

#### 2. Construcción OBT (Completa: 2015-2018)
 **La OBT se construyó para los 4 años:**
- Registros en `ANALYTICS.OBT_TRIPS`: **~550M registros**
- Incluye Yellow + Green unificados con todas las columnas derivadas

#### 3. Entrenamiento ML (Split temporal)
**El pipeline de ML usa todo el periodo disponible:**
- **Train:** 2015-2016 (2 años)
- **Validación:** 2017 (1 año)
- **Test:** 2018 (1 año)

**Escalable:** Si hay más recursos disponibles, el código está preparado para procesar 2015-2025 completo modificando únicamente las variables de ambiente en `.env`.

### Resumen de cobertura final

| Componente | Años cubiertos | Registros | Limitación |
|------------|----------------|-----------|------------|
| **RAW (Yellow/Green)** | 2015-2018 | ~550M |  Recursos computacionales |
| **OBT (analytics)** | 2015-2018 | ~550M | Tiempo de procesamiento |
| **ML Training** | 2015-2018 (split temporal) | ~200k |  Completo |

---

## Arquitectura

### Flujo de datos

```
1. Ingesta RAW
   ↓
   Descarga Parquet por mes/año → RAW.YELLOW_TAXIS + RAW.GREEN_TAXIS
   
2. Enriquecimiento
   ↓
   Carga catálogos → Unifica Y+G con JOINs → RAW.UNIFIED_TRIPS
   
3. OBT Construction
   ↓
   Agrega derivadas + temporales → ANALYTICS.OBT_TRIPS
   
4. Machine Learning
   ↓
   Preprocesamiento → Ensambles (Voting, Bagging, Boosting) → Selección del mejor
   
5. Evaluación
   ↓
   Métricas en Test + Diagnóstico (residuales, SHAP, importancias)
```

### Servicios Docker Compose

- **spark-notebook:** Jupyter + Spark para procesamiento de datos y ML (http://localhost:8888)
- **Spark UI:** Monitoreo de trabajos Spark (http://localhost:4040)

**Nota:** Snowflake se accede remotamente mediante credenciales en `.env`

---

## Estrategia de backfill & idempotencia

Durante el proceso se identificó que el volumen total de datos requería una estrategia de carga incremental controlada para evitar timeouts y problemas de memoria.

### Ingesta RAW (2015-2018)

#### Estrategia implementada

- **Ingesta mensual controlada con Spark:**
  Cada archivo Parquet representa un mes de viajes para un servicio (`yellow` o `green`).
  Se descarga con `requests` y se procesa con PySpark en chunks controlados.

- **Carga por chunks hacia Snowflake:**
  Se divide en lotes de ~1M de filas usando `write_pandas`.
  Cada chunk se exporta directamente a Snowflake, lo que permite manejar grandes volúmenes sin saturar memoria.

- **Columnas de control agregadas para trazabilidad:**
  - `INGESTED_AT_UTC`: Marca de tiempo de ingesta (UTC)
  - `CHUNK_ID`: Indica el número de chunk dentro de ese mes
  - `SOURCE_YEAR` y `SOURCE_MONTH`: Para auditoría y consultas históricas
  - `RUN_ID`: Identificador único de ejecución para rastrear procesos

- **Registro de auditoría y control de idempotencia:**
  - Cada mes procesado se registra en tablas de auditoría (`RAW.AUDIT_YELLOW` y `RAW.AUDIT_GREEN`) con el estado de la carga (`ok`, `fail`, `skip`)
  - Antes de cargar un mes se consulta esta tabla: si ya existe con estado `ok`, el proceso lo **omite automáticamente** para evitar duplicados
  - Se definió una **PRIMARY KEY** sobre campos clave para asegurar que no se inserten filas duplicadas si un proceso se vuelve a ejecutar

#### Estadísticas finales (RAW)
- **Yellow Taxi:** ~4900M viajes (2015-2018)
- **Green Taxi:** ~56M viajes (2015-2018)
- **Total:** ~550M registros

**Evidencia:** Archivos de auditoría en `auditoria/`

---

## Configuración del entorno

### Requisitos previos
- Docker y Docker Compose instalados
- Cuenta Snowflake activa con warehouse configurado
- 8GB+ RAM recomendado (16GB óptimo para ML)
- 20GB+ espacio en disco libre

### Variables de ambiente

#### Configuración `.env`

Copia `.env.example` a `.env` y configura tus credenciales:

```bash
# Snowflake Connection
SNOWFLAKE_ACCOUNT=tu_cuenta
SNOWFLAKE_USER=tu_usuario
SNOWFLAKE_PASSWORD=tu_contraseña
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=NYC_TAXI
SNOWFLAKE_ROLE=ACCOUNTADMIN

# Schemas
SNOWFLAKE_SCHEMA_RAW=RAW
SNOWFLAKE_SCHEMA_ANALYTICS=ANALYTICS

# Ingesta Parameters
START_YEAR=2015
END_YEAR=2018
SERVICES=yellow,green
BASE_URL=https://d37ci6vzurychx.cloudfront.net/trip-data
BATCH_SIZE=1000000

# Run Control
RUN_ID=auto  # auto-genera UUID, o especifica manualmente
```

#### Descripción de variables

| Variable | Propósito | Ejemplo |
|----------|-----------|---------|
| `SNOWFLAKE_ACCOUNT` | Identificador de cuenta Snowflake | `GTYEDVG-GHFTHF` |
| `SNOWFLAKE_USER` | Usuario con permisos de escritura | `data_engineer` |
| `SNOWFLAKE_PASSWORD` | Contraseña del usuario | `***` |
| `SNOWFLAKE_WAREHOUSE` | Warehouse para ejecutar queries | `COMPUTE_WH` |
| `SNOWFLAKE_DATABASE` | Database destino | `NYC_TAXI` |
| `SNOWFLAKE_ROLE` | Rol con permisos suficientes | `ACCOUNTADMIN` |
| `START_YEAR` | Año inicial de carga | `2015` |
| `END_YEAR` | Año final de carga | `2018` |
| `SERVICES` | Servicios a procesar (separados por coma) | `yellow,green` |
| `BATCH_SIZE` | Filas por batch en ingesta | `1000000` |

---

## Ejecución paso a paso

### 1. Levantar infraestructura

```bash
# Clonar repositorio
git clone <repo-url>
cd pset5-ensemble-ml

# Configurar variables de ambiente
cp .env.example .env
nano .env  # Editar con tus credenciales de Snowflake

# Levantar Docker Compose
docker-compose up -d

# Verificar contenedores
docker-compose ps
```

**Servicios disponibles:**
- Jupyter Notebook: http://localhost:8888 (sin token)
- Spark UI: http://localhost:4040

### 2. Ejecutar notebooks en orden

#### Notebook 01: Ingesta Parquet RAW (~4-6 horas para 2015-2018)

```python
# En Jupyter: notebooks/01_ingesta_parquet_raw.ipynb
# Este notebook:
# - Descarga Parquet mes a mes desde NYC TLC
# - Carga en chunks de 1M filas a Snowflake
# - Crea tablas RAW.YELLOW_TAXIS y RAW.GREEN_TAXIS
# - Registra auditoría en RAW.AUDIT_YELLOW y RAW.AUDIT_GREEN
# - Implementa idempotencia (salta meses ya cargados)
```

**Output esperado:**
- `RAW.YELLOW_TAXIS`
- `RAW.GREEN_TAXIS`
- Tablas de auditoría con estado por mes/año

#### Notebook 02: Enriquecimiento y Unificación (~1-2 horas)

```python
# notebooks/02_enriquecimiento_y_unificacion.ipynb
# Este notebook:
# - Carga catálogos (Taxi Zones, Payment Types, Rate Codes, Vendors)
# - Unifica Yellow + Green con UNION ALL
# - Hace JOINs con dimensiones
# - Crea RAW.UNIFIED_TRIPS (tabla intermedia enriquecida)
```

**Output esperado:**
- `RAW.UNIFIED_TRIPS`: ~364M registros con zonas y catálogos

#### Notebook 03: Construcción OBT (~30-40 min)

```python
# notebooks/03_construccion_obt.ipynb
# Este notebook:
# - Lee RAW.UNIFIED_TRIPS
# - Agrega columnas derivadas (trip_duration_min, avg_speed_mph, tip_pct)
# - Agrega columnas temporales (pickup_date, pickup_hour, day_of_week, etc.)
# - Crea ANALYTICS.OBT_TRIPS
# - Aplica PRIMARY KEY para idempotencia
```

**Output esperado:**
- `ANALYTICS.OBT_TRIPS`: ~550M registros con todas las métricas

#### Notebook 04: Machine Learning - Pipeline de Ensambles (~3-4 horas)

```python
# notebooks/pruebas_pset5.ipynb
# Este notebook:
# - Carga datos desde ANALYTICS.OBT_TRIPS
# - Preprocesamiento: imputación, escalado, encoding, polynomial features
# - Split temporal: Train (2015-2016), Val (2017), Test (2018)
# - Baseline: Media y Mediana
# - Voting Regressor: 3 modelos base (RF, GB, Ridge)
# - Bagging vs Pasting: con DecisionTree
# - Boosting: AdaBoost, GBDT, XGBoost (profundo), LightGBM, CatBoost
# - GridSearch/RandomSearch con TimeSeriesSplit
# - Selección del mejor modelo por RMSE en validación
# - Evaluación final en Test
# - Diagnóstico: residuales, error por buckets, SHAP, importancias
```

**Output esperado:**
- Tabla comparativa de todos los modelos (ver `evidences/` para resultados completos)
- Mejor modelo: **Gradient Boosting** con RMSE Test: **$3.5490**
- Gráficos de diagnóstico (residuales, importancias, SHAP)
- Análisis de error por distancia, hora, borough

---

## Diseño de esquemas

### Schema RAW

#### Tabla: `YELLOW_TAXIS`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `VENDORID` | NUMBER | ID del proveedor (1=CMT, 2=Curb, 6=Myle, 7=Helix) |
| `TPEP_PICKUP_DATETIME` | TIMESTAMP_NTZ | Fecha/hora de inicio del viaje |
| `TPEP_DROPOFF_DATETIME` | TIMESTAMP_NTZ | Fecha/hora de fin del viaje |
| `PASSENGER_COUNT` | NUMBER | Número de pasajeros |
| `TRIP_DISTANCE` | FLOAT | Distancia del viaje en millas |
| `RATECODEID` | NUMBER | Código de tarifa (1=Standard, 2=JFK, etc.) |
| `PULOCATIONID` | NUMBER | ID de zona de pickup |
| `DOLOCATIONID` | NUMBER | ID de zona de dropoff |
| `PAYMENT_TYPE` | NUMBER | Tipo de pago (0-6) |
| `FARE_AMOUNT` | FLOAT | Tarifa base |
| `TIP_AMOUNT` | FLOAT | Propina |
| `TOTAL_AMOUNT` | FLOAT | Total del viaje |
| `CONGESTION_SURCHARGE` | FLOAT | Recargo por congestión |
| `AIRPORT_FEE` | FLOAT | Tarifa aeropuerto |
| `RUN_ID` | STRING | UUID de la ejecución |
| `INGESTED_AT_UTC` | TIMESTAMP_NTZ | Timestamp de ingesta |
| `SOURCE_YEAR` | NUMBER | Año del dato fuente |
| `SOURCE_MONTH` | NUMBER | Mes del dato fuente |
| `CHUNK_ID` | VARCHAR(50) | Identificador de chunk |

**PRIMARY KEY:** `(TPEP_PICKUP_DATETIME, TPEP_DROPOFF_DATETIME, PULOCATIONID, DOLOCATIONID)`

### Schema ANALYTICS

#### Tabla: `OBT_TRIPS` (One Big Table)

**Grano:** 1 fila = 1 viaje

**Columnas (47 totales):**

##### Dimensión Temporal (9 columnas)
- `PICKUP_DATETIME`, `DROPOFF_DATETIME`
- `PICKUP_DATE`, `PICKUP_HOUR`
- `DAY_OF_WEEK`, `MONTH`, `YEAR`

##### Dimensión Geográfica (8 columnas)
- `PU_LOCATION_ID`, `PU_ZONE`, `PU_BOROUGH`, `PU_SERVICE_ZONE`
- `DO_LOCATION_ID`, `DO_ZONE`, `DO_BOROUGH`, `DO_SERVICE_ZONE`

##### Dimensión de Servicio (9 columnas)
- `SERVICE_TYPE` (yellow/green)
- `VENDOR_ID`, `VENDOR_NAME`
- `RATE_CODE_ID`, `RATE_CODE_DESC`
- `PAYMENT_TYPE`, `PAYMENT_TYPE_DESC`

##### Métricas de Viaje (3 columnas)
- `PASSENGER_COUNT`
- `TRIP_DISTANCE`
- `STORE_AND_FWD_FLAG`

##### Métricas Financieras (11 columnas)
- `FARE_AMOUNT`, `EXTRA`, `MTA_TAX`
- `TIP_AMOUNT`, `TOLLS_AMOUNT`
- `TOTAL_AMOUNT` (target)

##### Columnas Derivadas (3 columnas)
- `TRIP_DURATION_MIN` = `DATEDIFF(SECOND, PICKUP, DROPOFF) / 60`
- `AVG_SPEED_MPH` = `TRIP_DISTANCE / (DURATION / 60)`
- `TIP_PCT` = `(TIP_AMOUNT / FARE_AMOUNT) * 100`

**PRIMARY KEY:** `(PICKUP_DATETIME, DROPOFF_DATETIME, PU_LOCATION_ID, DO_LOCATION_ID, SERVICE_TYPE)`

---

## Machine Learning: Predicción de total_amount

### Prevención de data leakage

**Features NO permitidas (información post-viaje):**
-  `dropoff_datetime`
-  `trip_duration_min`
-  `avg_speed_mph`
-  `tip_pct`

**Features permitidas (disponibles en pickup):**
- `trip_distance` (estimado por GPS/taxímetro)
-  `pickup_hour`, `day_of_week`, `month`, `year`
-  `passenger_count`
-  `pu_borough`, `pu_zone`
-  `service_type`, `vendor_id`, `rate_code_id`
-  Flags derivados: `is_rush_hour`, `is_weekend`

### Preprocesamiento

```python
# 1. Imputación
SimpleImputer(strategy='median') para numéricas
SimpleImputer(strategy='most_frequent') para categóricas

# 2. Escalado (OBLIGATORIO para regularización)
StandardScaler() en numéricas

# 3. One-Hot Encoding
- Top-K + "Other" para controlar cardinalidad
- pu_borough: Top 5 + Other
- pu_zone: Top 20 + Other

# 4. Polynomial Features (grado 2)
- Solo en 2-3 numéricas clave: trip_distance, passenger_count
- Limitar combinaciones para evitar explosión de features
```

### Split temporal (no aleatorio)

```python
# Train: 2015-2016 (2 años, 66%)
# Validación: 2017 (1 año, 17%)
# Test: 2018 (1 año, 17%)
```

**Justificación:** Split temporal respeta la naturaleza secuencial de los datos y simula predicción en producción.

### Comparativa de ensambles

#### Experimentos realizados

1. **Baseline**
   - Media del target
   - Mediana del target

2. **Voting Regressor**
   - 3 modelos base: RandomForest, GradientBoosting, Ridge
   - Strategy: averaging

3. **Bagging vs Pasting**
   - Base learner: DecisionTreeRegressor
   - Comparación con `bootstrap=True/False`

4. **Boosting (5 algoritmos)**
   - AdaBoost
   - Gradient Boosting
   - **XGBoost** (estudio profundo)
   - LightGBM
   - CatBoost

### Resultados principales

 **MEJOR MODELO (por RMSE Validación):** Gradient Boosting

**Métricas en Test:**
- **RMSE:** $3.5490
- **MAE:** $1.853526
- **R²:** 0.932535

**Para ver resultados completos de todos los modelos:** Ver carpeta `evidences/` con tablas comparativas, gráficos y logs de GridSearch.

---
### Buenas prácticas

1. **Balance learning_rate vs n_estimators:**
   - Usar learning_rate bajo (0.01-0.05) con más árboles para mejor generalización
   - Relación inversa: menor LR requiere más estimators

2. **Early stopping:**
   - Usar conjunto de validación separado
   - Configurar `early_stopping_rounds=50` para detener si no hay mejora

3. **Manejo de categóricas:**
   - One-Hot Encoding con control de cardinalidad (Top-K + "Other")
   - Evitar target encoding sin validación cruzada (riesgo de leakage)

4. **Regularización progresiva:**
   - Empezar sin regularización (gamma=0, alpha=0, lambda=1)
   - Añadir gradualmente si hay overfitting
   - L2 (lambda) suele ser suficiente; L1 (alpha) para feature selection

5. **Validación temporal:**
   - No usar KFold tradicional en series temporales
   - Implementar TimeSeriesSplit o split por bloques

6. **Interpretabilidad:**
   - Usar `feature_importances_` para ranking de features
   - SHAP values para explicar predicciones individuales
   - Partial Dependence Plots para visualizar relaciones

### Casos de uso y pitfalls frecuentes

#### Casos de uso habituales
- Scoring crediticio
- Detección de fraude
- Sistemas de recomendación
- Predicción de demanda
- **Pricing dinámico** (como este proyecto)

#### Pitfalls frecuentes 

1. **Data leakage:**
   - Incluir variables calculadas con información futura
   - Usar `dropoff_datetime` para predecir `total_amount`
   - **Solución:** Feature engineering riguroso, validar temporalidad

2. **Target leakage en encoding:**
   - Target encoding usando todo el dataset (incluye test)
   - **Solución:** Calcular stats solo en train, aplicar en val/test

3. **Falta de validación temporal:**
   - Usar shuffle en series temporales
   - **Solución:** TimeSeriesSplit o split secuencial

4. **Overfitting por falta de regularización:**
   - Árboles muy profundos sin límites# proyecto_05
