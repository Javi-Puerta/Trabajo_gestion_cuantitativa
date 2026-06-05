# Trabajo de Gestión Cuantitativa

Repositorio del proyecto de **Gestión Cuantitativa**. La carpeta reúne la documentación final entregable, las presentaciones utilizadas durante el seguimiento del trabajo, el código del modelo de selección de activos, los notebooks de backtesting, los resultados guardados, la operativa enviada y el módulo de seguimiento/riesgo de la cartera.

El objetivo principal del proyecto es construir y evaluar una estrategia de inversión cuantitativa basada en **stock picking mediante modelos de Machine Learning**, aplicada sobre universos bursátiles como el EURO STOXX 50 y el S&P 500. El flujo completo incluye descarga de datos, construcción de variables, entrenamiento del modelo, selección de activos, optimización de pesos, backtesting histórico, generación de operativas y análisis posterior de resultados y riesgo.

---

## Estructura general del repositorio

```text
Trabajo_gestion_cuantitativa/
├── Memoria y presentaciones/        # Documentación final y presentaciones del proyecto
├── Backtest_series/                 # Series y resultados de backtests guardados en .pkl
├── Envios/                          # Operativas semanales enviadas y archivo histórico
├── Monitoring/                      # Notebooks y funciones para seguimiento y riesgo
├── mi_cartera/                      # Estado operativo de la cartera y operaciones realizadas
├── pruebas/                         # Pruebas y notebooks auxiliares de desarrollo
├── Aplicacion_modelo.ipynb          # Notebook de aplicación práctica del modelo
├── backtest_historico.ipynb         # Notebook principal de backtesting histórico
├── backtest_monos.ipynb             # Backtest aleatorio/de referencia tipo “monos”
├── *.py                             # Módulos principales del sistema de inversión
├── *_historico_cambios.csv          # Cambios históricos de composición de índices
├── latex_memoria.zip                # Proyecto LaTeX de la memoria
└── README.md                        # Este documento
```
---

## Carpetas principales

### `Memoria y presentaciones/`

Contiene los documentos finales y las presentaciones utilizadas durante el desarrollo del trabajo.

```text
Memoria y presentaciones/
├── Memoria proyecto.pdf
├── Presentacion 14 abril.pdf
├── Presentacion 6 mayo.pdf
├── Presentacion 14 mayo.pdf
└── Presentacion 10 junio.pdf
```

La **memoria** recoge la explicación completa de la metodología, el diseño del modelo, la construcción de la estrategia, los backtests, el análisis de resultados y las conclusiones. Las presentaciones muestran la evolución del proyecto en las distintas fechas de seguimiento.

---

### `Backtest_series/`

Guarda los resultados serializados de los backtests para evitar tener que recalcularlos cada vez.

```text
Backtest_series/
├── series_bt_from2020.pkl
├── series_bt_from2010.pkl
├── series_bt_crisis2008.pkl
├── series_bt_covid.pkl
└── backtest_monos.pkl
```

Estos archivos contienen las series de valor de cartera y resultados asociados a las distintas ventanas históricas analizadas. Se usan para comparar estrategias, generar tablas de métricas, representar gráficos y alimentar el análisis incluido en la memoria y en la presentación.

---

### `Envios/`

Contiene las operativas semanales del grupo y el histórico consolidado.

```text
Envios/
├── Operativa_Grupo5_12032026.xlsx
├── Operativa_Grupo5_20032026.xlsx
├── Operativa_Grupo5_27032026.xlsx
├── Operativa_Grupo5_03042026.xlsx
├── Operativa_Grupo5_10042026.xlsx
├── Operativa_Grupo5_17042026.xlsx
├── Operativa_Grupo5_24042026.xlsx
├── Operativa_Grupo5_01052026.xlsx
├── Operativa_Grupo5_08052026.xlsx
└── historico_operativa.xlsx
```

Cada archivo `Operativa_Grupo5_*.xlsx` representa una operativa semanal enviada. El archivo `historico_operativa.xlsx` recoge la evolución agregada de las decisiones realizadas a lo largo del proyecto.

---

### `Monitoring/`

Carpeta dedicada al seguimiento de la cartera y al análisis de riesgo.

```text
Monitoring/
├── auxfun.py
├── var_funciones.py
├── resultados.ipynb
└── var_tailvar.ipynb
```

Incluye funciones auxiliares para generar tablas, gráficos y métricas de seguimiento, además de notebooks orientados al análisis de resultados, cálculo de VaR, TailVaR y otras medidas de riesgo.

---

### `mi_cartera/`

Guarda el estado operativo de la cartera utilizada por el motor de inversión.

```text
mi_cartera/
├── cartera_actual.json
├── historial_operaciones.csv
├── modelo_estado.pkl
└── ultimo_entrenamiento.txt
```

Esta carpeta permite conservar la información necesaria para continuar la ejecución de la estrategia entre sesiones: composición actual, historial de compras y ventas, estado del modelo y fecha del último entrenamiento.

---

### `pruebas/`

Contiene notebooks y código utilizado para pruebas de desarrollo.

```text
pruebas/
├── StockPickingModel.py
├── backtest_3m.ipynb
├── prueba_StockPickingModel.ipynb
└── prueba_modelo_ML.ipynb
```

No constituye la parte principal de la entrega, pero documenta experimentos previos, validaciones parciales y pruebas realizadas antes de consolidar la arquitectura final.

---

## Notebooks principales

### `backtest_historico.ipynb`

Notebook principal para ejecutar y analizar los backtests históricos. Permite evaluar la estrategia en distintas ventanas temporales, comparar el comportamiento de los modelos y generar resultados utilizados en la memoria.

### `backtest_monos.ipynb`

Notebook de comparación con carteras aleatorias o estrategias de referencia. Se utiliza para contextualizar el desempeño del modelo frente a asignaciones no informadas.

### `Aplicacion_modelo.ipynb`

Notebook de aplicación práctica del modelo. Está orientado a ejecutar el flujo de selección de activos y generación de cartera en el contexto operativo del proyecto.

---

## Módulos Python principales

El código está organizado de forma modular para separar cada etapa del proceso de inversión.

```text
UniversoActivos.py
ProveedorDatos.py
VariablesTransformation.py
Modelos.py
Estrategia.py
Backtest.py
MotorInversion.py
auxiliary_functions.py
```

### `UniversoActivos.py`

Define la construcción del universo de inversión. Incluye clases para trabajar con universos estáticos y dinámicos, permitiendo reconstruir la composición histórica de índices a partir de archivos CSV de cambios.

### `ProveedorDatos.py`

Centraliza la descarga y preparación inicial de datos de mercado, principalmente mediante `yfinance`. Permite obtener precios diarios y semanales, volumen en dólares y dividendos.

### `VariablesTransformation.py`

Construye las variables explicativas utilizadas por el modelo. Incluye indicadores de momentum, volatilidad, drawdown, volumen, beta, RSI, MACD, medias móviles y bandas de Bollinger, además de la construcción del target de clasificación.

### `Modelos.py`

Contiene la lógica de entrenamiento de modelos de Machine Learning. Incluye modelos como Random Forest y XGBoost, junto con validación temporal mediante Walk-Forward Cross Validation.

### `Estrategia.py`

Define las estrategias de selección y asignación de pesos. Incluye estrategias equiponderadas, estrategias con optimización Monte Carlo y estrategias con componente macro o activo de cobertura.

### `Backtest.py`

Implementa el motor de backtesting. Simula la evolución histórica de la cartera, aplicando rebalanceos, costes, pesos objetivo y reglas de inversión.

### `MotorInversion.py`

Permite aplicar el modelo de forma operativa, manteniendo el estado de la cartera, registrando operaciones y actualizando el modelo cuando corresponde.

### `auxiliary_functions.py`

Agrupa funciones auxiliares de cálculo financiero, métricas, indicadores técnicos, costes, mark-to-market, generación de tablas y visualizaciones.

---

## Archivos de datos auxiliares

```text
eurostoxx50_historico_cambios.csv
sp500_historico_cambios.csv
ultima_fecha_dividendos.txt
```

Los archivos CSV recogen cambios históricos en la composición de los índices y permiten evitar sesgos de supervivencia al reconstruir el universo disponible en cada fecha. El archivo `ultima_fecha_dividendos.txt` se utiliza como referencia auxiliar para el tratamiento de dividendos.

---

## Flujo de trabajo del proyecto

El funcionamiento general del repositorio puede resumirse así:

1. **Definición del universo de inversión** mediante `UniversoActivos.py` y los archivos de cambios históricos.
2. **Descarga de datos de mercado** con `ProveedorDatos.py`.
3. **Construcción de variables explicativas y target** con `VariablesTransformation.py`.
4. **Entrenamiento del modelo** con `Modelos.py`.
5. **Selección de activos y asignación de pesos** con `Estrategia.py`.
6. **Simulación histórica** mediante `Backtest.py` y los notebooks de backtesting.
7. **Guardado de resultados** en `Backtest_series/`.
8. **Generación de operativa semanal** y actualización de `mi_cartera/`.
9. **Análisis de seguimiento y riesgo** desde `Monitoring/`.
10. **Documentación final** en `Memoria y presentaciones/`.

---

## Requisitos orientativos

El proyecto está desarrollado en Python y utiliza principalmente las siguientes librerías:

```text
pandas
numpy
scikit-learn
xgboost
yfinance
scipy
matplotlib
requests
openpyxl
jupyter
```

Para ejecutar los notebooks, se recomienda crear un entorno virtual e instalar las dependencias necesarias:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install pandas numpy scikit-learn xgboost yfinance scipy matplotlib requests openpyxl jupyter
```

---

## Reproducción de resultados

Para reproducir el análisis principal:

1. Abrir el repositorio en el entorno de trabajo.
2. Instalar las dependencias indicadas.
3. Ejecutar `backtest_historico.ipynb` para generar o revisar los backtests principales.
4. Ejecutar `backtest_monos.ipynb` para la comparación con estrategias aleatorias.
5. Consultar los archivos guardados en `Backtest_series/` para cargar resultados ya calculados.
6. Revisar `Monitoring/resultados.ipynb` y `Monitoring/var_tailvar.ipynb` para el seguimiento de resultados y riesgo.
7. Consultar la memoria y presentaciones en `Memoria y presentaciones/`.

---

## Contenido entregable

La entrega del proyecto queda recogida principalmente en:

- `Memoria y presentaciones/Memoria proyecto.pdf`: documento final del trabajo.
- `Memoria y presentaciones/Presentacion 10 junio.pdf`: presentación final.
- `backtest_historico.ipynb`: ejecución y análisis de backtests principales.
- `backtest_monos.ipynb`: comparación con carteras aleatorias.
- `Backtest_series/`: resultados guardados de las simulaciones.
- `Envios/`: operativas semanales generadas durante el proyecto.
- `Monitoring/`: análisis de seguimiento y riesgo.
- Módulos `.py` de la raíz: implementación del sistema cuantitativo.

---

## Observaciones

- Los archivos `.pkl` permiten reutilizar resultados sin repetir cálculos costosos.
- Las carpetas `__pycache__/` y `.DS_Store` son archivos generados automáticamente por Python/macOS y no son necesarios para la evaluación del proyecto.
- La carpeta `.git/`, si se entrega comprimida, contiene el historial interno del repositorio, pero no es necesaria para ejecutar el código.
- La carpeta `pruebas/` se conserva como apoyo al desarrollo, aunque la parte consolidada se encuentra en los módulos principales y notebooks de la raíz.

---

## Autores

Trabajo realizado por el **Grupo 5** para la asignatura de Gestión Cuantitativa.
