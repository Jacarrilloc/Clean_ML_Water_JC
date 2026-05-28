# 💧 Diagnóstico de Calidad del Agua en la India
### Procesamiento de Alto Volumen de Datos — PySpark · Keras · GeoPandas · Hadoop

> **Pontificia Universidad Javeriana**  
> Curso: Computación de Alto Desempeño  
> Autor: Julian Andres Carrillo Chiquisa

---

## 📋 Descripción del Proyecto

Este proyecto aplica técnicas de **Procesamiento de Alto Volumen de Datos (PAVD)** para diagnosticar y predecir la calidad del agua en los ríos de la India. A partir de datos fisicoquímicos y biológicos provenientes del portal oficial del gobierno indio (RiverIndia), se construye un pipeline completo que va desde la ingesta distribuida en un clúster Hadoop hasta la predicción con una red neuronal densa y la visualización georreferenciada por estado.

**Pregunta central:**  
> *¿Es posible predecir el Índice de Calidad del Agua (WQI) de los ríos de la India usando Machine Learning en un entorno distribuido con Spark?*

---

## 🎯 Objetivo

Implementar modelos de predicción utilizando **MLlib de PySpark** y redes neuronales con **Keras/TensorFlow**, explorando técnicas de IA en entornos de procesamiento distribuido de alto volumen de datos.

---

## 🗂️ Estructura del Notebook

El notebook `Clean_ML_Water_Julian_Carrillo.ipynb` sigue un pipeline de 9 secciones:

```
1. Importación de bibliotecas y configuración del entorno Spark
2. Carga de datos desde HDFS
3. Análisis y preparación de datos (EDA + limpieza)
4. Visualización de parámetros fisicoquímicos
5. Cálculo del Índice de Calidad del Agua (WQI)
6. Análisis del WQI por parámetro y distribución estadística
7. Visualización geográfica con GeoPandas (mapa por estado)
8. Histograma de WQI por estado
9. Modelo predictivo: Red Neuronal Densa con Keras
```

---

## 🔬 ¿Qué se hizo paso a paso?

### 1. Configuración del entorno Spark
Se inicializa `findspark` para localizar la instalación de Spark en el clúster y se crea una `SparkSession` con nombre de aplicación `Calidad_Agua_Carrillo`. El entorno corre sobre un **clúster Hadoop real** con NameNode en `hdfs://10.195.34.34:9000`.

### 2. Carga de datos desde HDFS
El archivo `waterquality.csv` se lee directamente desde HDFS usando la DataFrame API de Spark (`spark.read.format("csv")`), lo que permite que el procesamiento ocurra de forma distribuida en los nodos del clúster sin copiar los datos al driver.

### 3. Preprocesamiento y limpieza
- **Inspección de tipos**: al leer un CSV sin esquema, Spark infiere todo como `StringType`. Se verifican y corrigen los tipos.
- **Detección de nulos**: se usa `F.isNull()` + `F.isnan()` para cubrir tanto nulos SQL como NaN de punto flotante. Resultado: **cero valores nulos**.
- **Conversión de tipos**: todas las columnas numéricas se castean a `FloatType` con `.cast()`.
- **Reducción de dimensionalidad**: se elimina la columna `TOTAL_COLIFORM` por multicolinealidad con `FECAL_COLIFORM`.
- **Estadísticas descriptivas**: se calculan count, mean, stddev, min y max por cada columna.

### 4. Visualización de parámetros
Se recolectan los datos del clúster al driver con `.collect()` y se grafican series temporales comparativas de los parámetros: DO vs pH, BOD vs Nitratos. Esto permite detectar patrones de contaminación cruzada entre indicadores.

### 5. Cálculo del WQI (Water Quality Index)
Se implementa la fórmula estándar del Índice de Calidad del Agua:

```
WQI = Σ (wᵢ × qrᵢ)
```

Donde cada parámetro recibe un **rango de calidad discretizado (qr)** en 6 niveles según umbrales internacionales (OMS), y un **peso (w)** según su importancia:

| Parámetro | Peso (wᵢ) | Umbral óptimo |
|---|---|---|
| Oxígeno Disuelto (DO) | 0.281 | > 6.0 mg/L |
| Coliformes Fecales | 0.281 | < 5 UFC/100mL |
| Conductividad (COND) | 0.234 | < 75 µS/cm |
| pH | 0.165 | 6.5 – 8.5 |
| Nitratos/Nitritos (NN) | 0.028 | < 20 mg/L |
| DBO (BOD) | 0.009 | < 3 mg/L |

**Clasificación final por WQI:**

| Rango | Etiqueta |
|---|---|
| [0, 25) | 🟢 Excelente |
| [25, 50) | 🔵 Buena |
| [50, 75) | 🟠 Baja |
| [75, 100) | 🔴 Muy Baja |
| ≥ 100 | 🟥 Inadecuada |

### 6. Visualización geográfica (GeoPandas)
Se construye un mapa coroplético de la India coloreado por WQI mediante un pipeline de 6 pasos:
1. Extraer estados únicos del DataFrame Spark
2. Cargar el shapefile de la India con GeoPandas (`.shp + .dbf + .shx + .prj`)
3. Normalizar nombres de estados entre ambas fuentes (e.g. `NCT of Delhi → Delhi`, reemplazar `&`)
4. Merge outer join por columna `STATE` (Spark ↔ GeoPandas)
5. Imputar WQI nulo con la mediana del dataset
6. Graficar con coloración por cuantiles definidos por usuario (`scheme='userdefined'`)

Las etiquetas de WQI sobre el mapa se ajustan automáticamente para evitar superposición usando `adjustText`.

### 7. Histograma de WQI por estado
Gráfico de barras horizontal que permite comparar el índice WQI de cada estado de la India, identificando visualmente las regiones más y menos contaminadas.

### 8. Modelo predictivo — Red Neuronal Densa (Keras)
Se entrena una red neuronal completamente conectada para **predecir el valor numérico del WQI** a partir de los 6 rangos de calidad calculados (`qrPH`, `qrDO`, `qrCOND`, `qrBOD`, `qrNN`, `qrFecal`).

**Arquitectura:**
```
Input (6) → Dense(350, ReLU) → Dense(350, ReLU) → Dense(350, ReLU) → Dense(1, Linear)
```

**Configuración de entrenamiento:**
| Parámetro | Valor |
|---|---|
| Épocas | 200 |
| Batch size | 81 |
| Optimizador | Adam (lr=0.001) |
| Función de pérdida | MSE |
| Split entrenamiento/prueba | 80% / 20% |
| Semilla aleatoria | `random_state=1` |

> **Nota técnica:** dado que Keras no trabaja directamente con DataFrames de Spark, los datos se convierten al driver con `.toPandas()` antes de entrenar.

---

## 📦 Librerías utilizadas

| Librería | Versión recomendada | Propósito |
|---|---|---|
| `pyspark` | ≥ 3.x | Motor de procesamiento distribuido (DataFrame API, SQL, MLlib) |
| `findspark` | ≥ 2.x | Localizar instalación de Spark en el sistema/clúster |
| `numpy` | ≥ 1.21 | Operaciones numéricas y arreglos |
| `pandas` | ≥ 1.3 | Manipulación tabular en el driver local |
| `matplotlib` | ≥ 3.4 | Visualización de gráficas y series |
| `seaborn` | ≥ 0.11 | Estilos visuales sobre matplotlib |
| `geopandas` | ≥ 0.10 | Carga y manipulación de datos geoespaciales (shapefile) |
| `mapclassify` | ≥ 2.4.0 | Esquemas de clasificación para mapas coropléticos |
| `adjustText` | ≥ 0.7 | Ajuste automático de etiquetas en mapas sin superposición |
| `keras` | ≥ 2.x / TF 2.x | Construcción y entrenamiento de la red neuronal |
| `tensorflow` | ≥ 2.x | Backend de Keras |
| `scikit-learn` | ≥ 0.24 | División train/test con `train_test_split` |

### Instalación de dependencias

```bash
pip install pyspark findspark numpy pandas matplotlib seaborn geopandas mapclassify adjustText keras tensorflow scikit-learn
```

> **Requisito de entorno:** el notebook está diseñado para correr sobre un **clúster Hadoop/Spark**. Para ejecución local, cambiar la ruta de carga del CSV de HDFS a una ruta local y ajustar la `SparkSession`.

---

## 📊 Dataset

| Campo | Detalle |
|---|---|
| **Fuente** | Portal oficial del gobierno de la India — RiverIndia |
| **Archivo** | `waterquality.csv` |
| **Almacenamiento** | HDFS — `hdfs://10.195.34.34:9000/csv/waterquality.csv` |
| **Cobertura** | +500 estaciones de monitoreo en ríos de la India |
| **Variables** | `STATION_CODE`, `LOCATIONS`, `STATE`, `TEMP`, `DO`, `pH`, `CONDUCTIVITY`, `BOD`, `NITRATE_N_NITRITE_N`, `FECAL_COLIFORM` |

---

## 🗺️ Recursos geoespaciales

Para la visualización en mapa se requiere el shapefile de los estados de la India:
- `india_states.shp`
- `india_states.dbf`
- `india_states.shx`
- `india_states.prj`

---

## 📈 Resultados principales

- **Estados con mejor calidad:** Kerala y Tamil Nadu (WQI < 25 — Excelente)
- **Estados con peor calidad:** Bihar y Maharashtra (WQI > 50 — Baja)
- **Factor más crítico:** Coliformes fecales en el cinturón nororiental
- **Variable de mayor peso en el WQI:** Oxígeno Disuelto (DO, w=0.281)
- **Comportamiento del modelo:** La curva de pérdida converge correctamente sin signos de sobreajuste en las épocas iniciales

---

## 🔭 Trabajo futuro

- Incorporar **series de tiempo** (datos mensuales/anuales) para capturar variaciones estacionales
- Implementar **k-fold cross-validation** para estimar la varianza del error
- Ajustar hiperparámetros con **Keras Tuner** o GridSearch
- Comparar contra modelos adicionales: **Random Forest**, **Gradient Boosting**, regresión múltiple

---

## 📚 Referencias

1. Sutadian, A. D. et al. (2016). *Development of river water quality indices — a review*. Ecological Indicators. [https://www.intechopen.com/chapters/69568](https://www.intechopen.com/chapters/69568)
2. Apache Software Foundation. *Apache Spark Documentation — PySpark SQL & DataFrames*. [https://spark.apache.org/docs/latest/api/python/](https://spark.apache.org/docs/latest/api/python/)
3. GeoPandas Development Team. *GeoPandas: Python tools for geographic data*. [https://geopandas.org/](https://geopandas.org/)
4. Chollet, F. et al. *Keras: The Python Deep Learning Library*. [https://keras.io/](https://keras.io/)