# Trabajo_gestion_cuantitativa

Git para guardar el código del modelo de stock-picking para el proyecto de gestión cuantitativa.

Buenas prácticas:
- Hacer commits frecuentes y explicando bien los cambios/aportaciones implementados.
- Mantener un esquema actualizado.
- Separar los cambios de backtest, motor operativo y funciones auxiliares.
- Evitar modificar varias piezas críticas a la vez sin comprobar resultados intermedios.

Esquema clases:
=== ARQUITECTURA DEL SISTEMA ===

FICHEROS
├── auxiliary_functions.py
├── UniversoActivos.py
├── ProveedorDatos.py
├── VariablesTransformation.py
├── Modelos.py
├── Estrategia.py
├── Backtest.py
├── MotorInversion.py
└── auxfun.py

─────────────────────────────────────────────────────────────

auxiliary_functions.py
│
├── calculate_rsi(series, period=14)
│   └── Calcula el RSI con ventana rolling de ganancias y pérdidas.
│
├── calculate_macd(series)
│   └── Calcula el MACD como diferencia entre EMA(12) y EMA(26).
│
├── calculate_bollinger(prices, period=20, num_std=2.0)
│   └── Devuelve las bandas superior e inferior de Bollinger.
│
├── calculate_beta(returns, market_returns, period)
│   └── Calcula beta rolling: cov(activo, mercado) / var(mercado).
│
├── compute_performance_metrics(level_series, periods_per_year=252, rf_annual=0.0)
│   └── Calcula rentabilidad total, rentabilidad anualizada, volatilidad, Sharpe,
│       Sortino, max drawdown, Calmar, win rate, mejor periodo y peor periodo.
│
├── build_metrics_table(series_dict, periods_per_year=252, rf_annual=0.0)
│   └── Recibe {nombre: serie_nivel} y devuelve una tabla comparativa de métricas.
│
├── calcular_costes(tickers) → dict
│   └── Devuelve un coste fijo por ticker. Actualmente 0.05% por operación.
│
├── mark_to_market(cartera, datos_hoy) → float
│   └── Calcula el valor total de la cartera como cash + posiciones a precios actuales.
│
├── _rolling_std(x, window)
│   └── Desviación típica rolling con min_periods = max(1, window//2).
│
└── _clip_by_quantiles(df, col, low_q=0.01, high_q=0.99)
    └── Winsoriza una columna por ticker según cuantiles.

─────────────────────────────────────────────────────────────

UniversoActivos.py
│
├── UniversoActivosBase (ABC)
│   ├── get_full_ticker_list() → list[str]
│   └── get_universe_at_date(date) → set[str]
│
├── UniversoActivosEstatico(tickers)
│   ├── get_full_ticker_list()
│   │   └── Devuelve siempre la misma lista de tickers.
│   └── get_universe_at_date(date)
│       └── Devuelve siempre el mismo universo.
│
└── UniversoActivosDinamico(tickers_actuales, start_date, end_date, csv_cambios_path)
    ├── _load_changes()
    │   └── Carga CSV con columnas [date, Tickr added, Tickr removed].
    ├── get_full_ticker_list()
    │   └── Devuelve tickers actuales + tickers eliminados históricamente.
    └── get_universe_at_date(date)
        └── Reconstruye el universo válido en esa fecha aplicando los cambios en orden inverso.

─────────────────────────────────────────────────────────────

ProveedorDatos.py
│
├── ProveedorDatosBase (ABC)
│   ├── download_prices_daily(tickers, start_date, end_date)
│   │   └── Interfaz para datos diarios.
│   └── download_prices_weekly(tickers, start_date, end_date)
│       └── Interfaz para datos semanales.
│
└── YFinanceProvider()
    ├── download_prices_daily(tickers, start_date, end_date)
    │   ├── Descarga datos diarios con yfinance.
    │   ├── Usa auto_adjust=False y actions=True.
    │   ├── Devuelve columnas:
    │   │   ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD', 'Dividendos']
    │   ├── Calcula Volumen_USD = Precio_Close * Volumen.
    │   ├── Rellena precios y volúmenes con ffill por ticker.
    │   └── Rellena Dividendos con 0.0 cuando no hay pago.
    │
    └── download_prices_weekly(tickers, start_date, end_date)
        ├── Parte de los datos diarios.
        ├── Remuestrea a W-FRI.
        ├── Precio_Close: último precio de la semana.
        ├── Volumen_USD: suma semanal.
        └── Dividendos: suma semanal.

─────────────────────────────────────────────────────────────

VariablesTransformation.py
│
└── FeatureEngineer(criterio, ticker_indice)
    │
    ├── feature_cols
    │   └── Lista de columnas usadas como variables explicativas.
    │
    ├── _build_daily_features(df_daily)
    │   ├── Calcula indicadores diarios por ticker.
    │   ├── RSI_14.
    │   ├── MACD.
    │   ├── Log_Precio_SMA_50.
    │   ├── Log_Precio_SMA_200.
    │   ├── Log_Precio_EMA_50.
    │   ├── Log_Precio_EMA_200.
    │   └── Remuestrea los indicadores al último valor semanal W-FRI.
    │
    └── build(df_weekly, df_daily)
        ├── Une variables semanales y variables diarias remuestreadas.
        ├── Incorpora Dividendos si existen; si no, los rellena con 0.0.
        │
        ├── Variables semanales:
        │   ├── Retorno_1W = (Precio_Close + Dividendos) / precio_prev - 1.
        │   ├── Momentum_12M.
        │   ├── Momentum_6M.
        │   ├── Momentum_1M.
        │   ├── Mom_12m_ex_1m.
        │   ├── RetRel_SPY_3m.
        │   ├── Volatilidad_6M.
        │   ├── Volatilidad_1M.
        │   ├── Vol_ratio_1m_6m.
        │   ├── DD_6m.
        │   ├── VolumenUSD_z_20.
        │   ├── Turnover_20w.
        │   ├── Retorno_t1.
        │   ├── Retorno_t2.
        │   ├── Ratio_52w_High.
        │   ├── Beta_12M.
        │   ├── Log_Precio_Boll_Upper.
        │   └── Log_Precio_Boll_Lower.
        │
        ├── Target:
        │   ├── criterio='mediana':
        │   │   └── Target = 1 si el retorno de la semana siguiente supera la mediana.
        │   └── criterio=N:
        │       └── Target = 1 si el activo está entre los N mejores retornos de la semana siguiente.
        │
        ├── Imputación:
        │   ├── Forward fill por ticker para evitar mirar al futuro.
        │   └── fillna(0) como último recurso.
        │
        └── Devuelve DataFrame con precios, variables, retorno semanal y target.

─────────────────────────────────────────────────────────────

Modelos.py
│
├── ModeloBase (ABC)
│   ├── train(df, feature_cols)
│   │   ├── Elimina filas sin Target.
│   │   ├── Entrena con GridSearchCV.
│   │   ├── Usa WalkForwardCV como validación temporal.
│   │   ├── Optimiza scoring='roc_auc'.
│   │   └── Guarda el mejor estimador en self.clf.
│   │
│   └── predict_proba(X)
│       └── Devuelve la probabilidad de clase positiva.
│
├── RandomForestModel(random_state=42, n_splits=5)
│   └── Modelo Random Forest con grid de:
│       ├── n_estimators.
│       ├── max_depth.
│       ├── min_samples_leaf.
│       ├── max_features.
│       └── class_weight.
│
├── XGBoostModel(random_state=42, n_splits=5)
│   └── Modelo XGBoost con grid de:
│       ├── n_estimators.
│       ├── max_depth.
│       ├── learning_rate.
│       ├── subsample.
│       ├── colsample_bytree.
│       └── scale_pos_weight.
│
└── WalkForwardCV(n_splits=3, val_ratio=0.20)
    ├── Cross-validation temporal.
    ├── Entrena desde el inicio de la muestra.
    ├── Valida en la ventana temporal inmediatamente posterior.
    └── Evita mezclar observaciones futuras en el entrenamiento.

─────────────────────────────────────────────────────────────

Estrategia.py
│
├── EstrategiaBase (ABC)
│   ├── train(df, feature_cols, tickers_validos, df_daily) → bool
│   │   ├── Filtra el entrenamiento a tickers válidos.
│   │   ├── Devuelve False si no hay datos.
│   │   └── Entrena self.modelo.
│   │
│   └── seleccionar(df_hoy, feature_cols, cartera, df_daily) → dict{ticker: peso}
│       └── Interfaz común para generar pesos objetivo.
│
├── EstrategiaMLEquiponderada(modelo, n_activos_obj, umbral_salida)
│   └── seleccionar()
│       ├── Calcula Score = modelo.predict_proba().
│       ├── Ordena activos por Score descendente.
│       ├── Mantiene activos ya en cartera si siguen dentro del top umbral_salida.
│       ├── Completa huecos con nuevos candidatos mejor puntuados.
│       └── Asigna pesos equiponderados.
│
├── EstrategiaMLMonteCarlo(modelo, n_activos_obj, umbral_salida,
│                          peso_max, n_simulaciones, peso_min, dias_retorno)
│   └── seleccionar()
│       ├── Preselecciona candidatos igual que EstrategiaMLEquiponderada.
│       ├── Construye retornos diarios históricos usando precios + dividendos.
│       ├── Simula carteras aleatorias con restricciones de peso mínimo y máximo.
│       ├── Calcula retorno anualizado, volatilidad anualizada y Sharpe.
│       └── Devuelve la cartera con mayor Sharpe simulado.
│
├── EstrategiaMLEquiponderadaMacro(modelo, n_activos_obj, umbral_salida,
│                                  ticker_indice, exposicion_rv, ticker_hedge)
│   ├── Hereda de EstrategiaMLEquiponderada.
│   ├── _señal_riesgo(fecha_hoy, df_daily)
│   │   └── Activa señal si el precio del índice está por debajo de su media móvil de 200 días.
│   └── seleccionar()
│       ├── Selecciona primero cartera de renta variable con la estrategia base.
│       ├── Si hay señal de riesgo, reduce la exposición a renta variable.
│       └── Asigna el peso restante al activo hedge.
│
└── EstrategiaMLMinVarAlphaTilt(modelo, n_activos_obj, umbral_salida, ...)
    ├── train()
    │   ├── Entrena el modelo de alpha.
    │   └── Estima matriz de covarianzas robusta con Ledoit-Wolf.
    │
    └── seleccionar()
        ├── Calcula alpha a partir del score del modelo.
        ├── Usa candidatos del top umbral_salida y posiciones actuales.
        ├── Optimiza pesos mediante SLSQP.
        ├── Función objetivo:
        │   └── - alpha·w + lambda_risk·w'Σw + lambda_tc·|w - w_prev|.
        ├── Impone long-only y peso máximo por activo.
        ├── Aplica no-trade band.
        ├── Aplica límite de turnover total.
        ├── Evalúa si la mejora esperada compensa costes.
        └── Devuelve los pesos finales de la cartera.

─────────────────────────────────────────────────────────────

Backtest.py
│
├── BacktestEngine(universo, proveedor, feature_engineer, estrategia,
│                  start_date, end_date, len_ventana, nominal)
│   │
│   ├── __init__()
│   │   ├── Ajusta start_date al primer viernes disponible.
│   │   ├── Obtiene tickers invertibles del universo.
│   │   ├── Añade ticker_indice al conjunto de descarga.
│   │   ├── Descarga df_daily y df_weekly desde len_ventana + 1 años antes.
│   │   └── Calcula costes por ticker.
│   │
│   ├── _datos_asof(fecha)
│   │   ├── Filtra df_daily y df_weekly hasta fecha.
│   │   ├── Construye features solo con información disponible hasta esa fecha.
│   │   └── Reduce el riesgo de look-ahead bias.
│   │
│   ├── _train(fecha_pivote, tickers_validos, df_asof, df_daily_asof) → bool
│   │   ├── Define fecha_corte = fecha_pivote - 2 semanas.
│   │   ├── Usa ventana histórica de len_ventana años.
│   │   ├── Filtra por tickers válidos.
│   │   └── Entrena la estrategia.
│   │
│   ├── _ajustar_pesos(cartera, precios_hoy) → dict{ticker: peso_real}
│   │   └── Calcula pesos reales de cartera a precios actuales.
│   │
│   ├── _cobrar_dividendos(cartera, datos_hoy) → float
│   │   ├── Calcula dividendos cobrados por las posiciones existentes.
│   │   └── Los suma al cash de la cartera.
│   │
│   ├── _ajustar_cartera(cartera, datos_hoy, pesos_nuevos)
│   │   ├── Ejecuta ventas de activos que salen de cartera.
│   │   ├── Ejecuta compras de activos nuevos.
│   │   ├── Ajusta posiciones que permanecen en cartera.
│   │   ├── Aplica costes de transacción.
│   │   ├── Evita posiciones negativas.
│   │   └── Devuelve cartera, pesos ajustados y valor total.
│   │
│   ├── _run() → DataFrame [Fecha, Valor cartera]
│   │   ├── Loop diario entre start_date y end_date.
│   │   ├── Cobra dividendos diariamente.
│   │   ├── Valora cartera con mark_to_market.
│   │   ├── Rebalancea en fechas semanales.
│   │   ├── Reentrena cada 180 días aproximadamente.
│   │   ├── Genera pesos objetivo con estrategia.seleccionar().
│   │   └── Guarda la evolución neta de la cartera.
│   │
│   └── print_results(bmks=None, bmk_equal_weight=None, plot=True)
│       ├── Ejecuta _run().
│       ├── Normaliza la serie de la estrategia.
│       ├── Compara contra benchmarks si se indican.
│       ├── Puede construir benchmark equiponderado con dividendos.
│       ├── Grafica evolución de cartera y benchmarks.
│       └── Devuelve fechas, serie de estrategia y tabla de métricas.
│
└── BacktestRandom(universo, proveedor, start_date, end_date, nominal, n_activos=15)
    ├── Backtest rápido de estrategias aleatorias.
    ├── Elige n_activos al azar cada semana.
    ├── Asigna pesos aleatorios con restricciones de peso mínimo y máximo.
    ├── Aplica costes según turnover semanal.
    ├── run_montecarlo(n_sims, benchmark)
    │   ├── Ejecuta muchas simulaciones aleatorias.
    │   ├── Devuelve media, std, p10, p50, p90.
    │   ├── Devuelve también todas las simulaciones.
    │   └── Puede añadir benchmark.
    └── Sirve como prueba de robustez frente a estrategias aleatorias.

─────────────────────────────────────────────────────────────

MotorInversion.py
│
└── MotorInversion(universo, feature_engineer, estrategia, estado_path,
                   len_ventana, capital_total, proveedor_cls)
    │
    ├── Archivos persistidos:
    │   ├── cartera_actual.json
    │   ├── historial_operaciones.csv
    │   ├── ultimo_entrenamiento.txt
    │   ├── modelo_estado.pkl
    │   └── ultima_fecha_dividendos.txt
    │
    ├── __init__()
    │   ├── Carga universo, costes, feature engineer y estrategia.
    │   ├── Crea carpeta de estado si no existe.
    │   ├── Carga cartera actual.
    │   ├── Lee fecha de último entrenamiento.
    │   ├── Lee fecha de últimos dividendos procesados.
    │   └── Carga modelo entrenado si existe.
    │
    ├── ejecutar(fecha) → DataFrame de señales
    │   ├── Reentrena si corresponde.
    │   ├── Genera señales.
    │   ├── Guarda cartera.
    │   ├── Guarda historial de operaciones.
    │   ├── Guarda modelo.
    │   └── Devuelve tabla de señales.
    │
    ├── _reentrenar_si_toca(fecha)
    │   ├── Reentrena si no hay modelo previo.
    │   ├── Reentrena si han pasado MESES_RETRAIN * 30 días.
    │   ├── Descarga datos históricos.
    │   ├── Construye features.
    │   ├── Corta el entrenamiento dos semanas antes de la fecha actual.
    │   └── Guarda fecha de último entrenamiento.
    │
    ├── _descargar_datos(fecha, long_hist)
    │   ├── Descarga desde fecha - (long_hist + 1) años.
    │   ├── Descarga hasta fecha + 1 día.
    │   ├── Incluye tickers de cartera/universo.
    │   └── Incluye ticker del índice.
    │
    ├── _leer_fecha_dividendos() / _guardar_fecha_dividendos(fecha)
    │   └── Controla hasta qué fecha se han incorporado dividendos al cash.
    │
    ├── _actualizar_dividendos(fecha, df_daily) → float
    │   ├── Comprueba que df_daily tenga columna Dividendos.
    │   ├── Calcula dividendos entre última fecha procesada y fecha actual.
    │   ├── Multiplica dividendos por acciones en cartera.
    │   ├── Suma el importe cobrado al cash.
    │   └── Actualiza ultima_fecha_dividendos.txt.
    │
    ├── _generar_señales(fecha) → DataFrame
    │   ├── Descarga datos actualizados.
    │   ├── Actualiza dividendos.
    │   ├── Construye features.
    │   ├── Filtra tickers válidos en la fecha.
    │   ├── Valora la cartera antes de operar.
    │   ├── Llama a estrategia.seleccionar().
    │   ├── Genera órdenes de VENTA para activos que salen.
    │   ├── Genera órdenes de COMPRA para activos nuevos.
    │   ├── Ajusta posiciones de activos que permanecen.
    │   ├── Aplica costes de transacción.
    │   └── Devuelve columnas:
    │       [Ticker, Accion, Cantidad, Precio, CT, Precio_Ejecutado]
    │
    ├── _cargar_cartera() / _guardar_cartera()
    │   └── Lee y escribe cartera_actual.json.
    │
    ├── _guardar_historial(fecha, señales)
    │   └── Añade las señales al CSV historial_operaciones.csv.
    │
    ├── _leer_fecha_train()
    │   └── Lee ultimo_entrenamiento.txt.
    │
    └── _cargar_modelo() / _guardar_modelo()
        └── Serializa o carga el modelo con pickle.

─────────────────────────────────────────────────────────────

auxfun.py
│
├── Funciones auxiliares para análisis posterior de resultados reales.
│
├── get_eurostoxx50_tickers()
│   └── Obtiene tickers actuales del EURO STOXX 50 desde Wikipedia.
│
├── historico_valor_cartera(...)
│   ├── Reconstruye la evolución diaria de la cartera real desde operaciones.
│   ├── Tiene en cuenta compras, ventas, costes y dividendos.
│   └── Devuelve cash, valor de acciones, dividendos diarios, valor cartera y rentabilidad diaria.
│
├── pnl_por_activo(...)
│   ├── Calcula P&L neto por activo.
│   ├── Incluye operaciones, posición final y dividendos.
│   └── Genera gráfico de barras horizontales.
│
├── tabla_semanal_atribucion(...)
│   ├── Calcula atribución semanal.
│   ├── Separa benchmark, selección, pesos y costes.
│   └── Trabaja con retornos semanales incluyendo dividendos.
│
├── encadenar_atribucion(...)
│   ├── Encadena los efectos semanales.
│   ├── Convierte efectos porcentuales en euros.
│   └── Comprueba que la descomposición cuadra con el NAV.
│
├── formatear_atribucion(...)
│   └── Da formato a tablas semanales y finales.
│
├── series_diarias_cartera_bmks(...)
│   ├── Reconstruye serie diaria de cartera real.
│   ├── Construye benchmark.
│   ├── Calcula métricas.
│   └── Devuelve series, tablas y métricas.
│
└── grafico_evolucion_drawdown(...)
    └── Grafica evolución de cartera y drawdown.

─────────────────────────────────────────────────────────────

FLUJO DE DATOS DEL MODELO

1. Definición del universo
│
│  tickers actuales + csv de cambios históricos
│       ↓
│  UniversoActivos
│       ↓
│  tickers válidos en cada fecha
│
└─────────────────────────────────────────────────────────────

2. Descarga de datos
│
│  tickers válidos + ticker índice
│       ↓
│  YFinanceProvider
│       ↓
│  df_daily:
│      Fecha, Ticker, Precio_Close, Volumen_USD, Dividendos
│
│  df_weekly:
│      Fecha, Ticker, Precio_Close, Volumen_USD, Dividendos
│
└─────────────────────────────────────────────────────────────

3. Construcción de variables
│
│  df_daily + df_weekly
│       ↓
│  FeatureEngineer
│       ↓
│  DataFrame con:
│      precios, retornos, features, target
│
└─────────────────────────────────────────────────────────────

4. Entrenamiento
│
│  features históricas + target
│       ↓
│  ModeloBase / RandomForestModel / XGBoostModel
│       ↓
│  modelo entrenado mediante WalkForwardCV
│
└─────────────────────────────────────────────────────────────

5. Selección de cartera
│
│  features de la fecha actual + modelo entrenado + cartera actual
│       ↓
│  Estrategia
│       ↓
│  pesos objetivo {ticker: peso}
│
└─────────────────────────────────────────────────────────────

6A. Backtest
│
│  pesos objetivo + precios + dividendos + costes
│       ↓
│  BacktestEngine
│       ↓
│  evolución histórica de cartera, métricas y gráficos
│
└─────────────────────────────────────────────────────────────

6B. Motor real
│
│  pesos objetivo + cartera_actual.json + precios actuales
│       ↓
│  MotorInversion
│       ↓
│  señales de compra/venta/mantener
│       ↓
│  cartera_actual.json + historial_operaciones.csv
│
└─────────────────────────────────────────────────────────────

USO (Backtest)

from UniversoActivos import UniversoActivosDinamico
from ProveedorDatos import YFinanceProvider
from VariablesTransformation import FeatureEngineer
from Modelos import RandomForestModel
from Estrategia import EstrategiaMLEquiponderada
from Backtest import BacktestEngine

universo = UniversoActivosDinamico(
    tickers_actuales=tickers,
    start_date=start_date,
    end_date=end_date,
    csv_cambios_path="eurostoxx50_historico_cambios.csv"
)

proveedor = YFinanceProvider()
fe = FeatureEngineer(criterio=15, ticker_indice="^STOXX50E")
modelo = RandomForestModel()
estrategia = EstrategiaMLEquiponderada(
    modelo=modelo,
    n_activos_obj=15,
    umbral_salida=22
)

engine = BacktestEngine(
    universo=universo,
    proveedor=proveedor,
    feature_engineer=fe,
    estrategia=estrategia,
    start_date=start_date,
    end_date=end_date,
    len_ventana=4,
    nominal=10_000_000
)

fechas, serie, metricas = engine.print_results(
    bmks=["^STOXX50E"],
    bmk_equal_weight=tickers,
    plot=True
)

─────────────────────────────────────────────────────────────

USO (Motor en producción, ejecución semanal)

from datetime import date
from UniversoActivos import UniversoActivosEstatico
from ProveedorDatos import YFinanceProvider
from VariablesTransformation import FeatureEngineer
from Modelos import RandomForestModel
from Estrategia import EstrategiaMLMonteCarlo
from MotorInversion import MotorInversion

universo = UniversoActivosEstatico(tickers)
fe = FeatureEngineer(criterio=15, ticker_indice="^STOXX50E")
modelo = RandomForestModel()

estrategia = EstrategiaMLMonteCarlo(
    modelo=modelo,
    n_activos_obj=15,
    umbral_salida=22,
    peso_max=0.20,
    n_simulaciones=5000,
    peso_min=0.02,
    dias_retorno=252
)

motor = MotorInversion(
    universo=universo,
    feature_engineer=fe,
    estrategia=estrategia,
    estado_path="./mi_cartera",
    len_ventana=4,
    capital_total=10_000_000,
    proveedor_cls=YFinanceProvider
)

señales = motor.ejecutar(date(2026, 3, 20))