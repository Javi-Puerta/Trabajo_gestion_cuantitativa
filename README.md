# Trabajo_gestion_cuantitativa
Git para guardar el código del modelo de stock-picking para el proyecto de gestión cuantitativa

Buenas prácticas:
- Hacer commits frecuentes y explicando bien los cambios/aportaciones implementados
- Mantener un esquema actualizado

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
└── MotorInversion.py

─────────────────────────────────────────────────────────────

auxiliary_functions.py
│
├── calculate_rsi(series, period=14)
│ └── RSI con ventana rolling de medias de ganancias/pérdidas
│
├── calculate_macd(series)
│ └── Diferencia entre EMA(12) y EMA(26)
│
├── calculate_bollinger(prices, period=20, num_std=2.0)
│ └── Devuelve (upper_band, lower_band)
│
├── calculate_beta(returns, market_returns, period)
│ └── Beta rolling: cov(activo, mercado) / var(mercado)
│
├── compute_performance_metrics(level_series, periods_per_year=252, rf_annual=0.0)
│ └── Calcula retorno total/anualizado, vol, Sharpe, Sortino, Max DD, Calmar, win rate
│
├── build_metrics_table(series_dict, periods_per_year=252, rf_annual=0.0)
│ └── Recibe {nombre: serie_nivel} y devuelve tabla comparativa de métricas
│
├── calcular_costes(tickers) → dict
│ └── Devuelve coste fijo (0.05%) por ticker para usar en backtest y motor
│
├── mark_to_market(cartera, datos_hoy) → float
│ └── Calcula el valor total de la cartera (cash + posiciones) a precios actuales
│
├── _rolling_std(x, window)
│ └── Std rolling con min_periods = max(1, window//2)
│
└── _clip_by_quantiles(df, col, low_q=0.01, high_q=0.99)
    └── Winsoriza una columna por ticker según cuantiles

─────────────────────────────────────────────────────────────

UniversoActivos.py
│
├── UniversoActivosBase (ABC)
│ ├── get_full_ticker_list() → list[str]
│ └── get_universe_at_date(date) → set[str]
│
├── UniversoActivosEstatico(tickers)
│ ├── get_full_ticker_list() → devuelve siempre la misma lista
│ └── get_universe_at_date(date) → devuelve siempre el mismo set
│
└── UniversoActivosDinamico(tickers_actuales, start_date, end_date, csv_cambios_path)
    ├── _load_changes() → carga CSV con columnas [date, Tickr added, Tickr removed]
    ├── get_full_ticker_list() → tickers actuales + todos los históricos (para descargar precios)
    └── get_universe_at_date(date) → reconstruye el universo en esa fecha aplicando cambios en orden inverso

─────────────────────────────────────────────────────────────

ProveedorDatos.py
│
├── ProveedorDatosBase (ABC)
│ ├── download_prices_daily() → DataFrame diario
│ └── download_prices_weekly() → DataFrame semanal
│
└── YFinanceProvider()   ← sin argumentos en el constructor
    ├── download_prices_daily(tickers, start_date, end_date)
    │ └── ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD'] con ffill para días sin datos
    └── download_prices_weekly(tickers, start_date, end_date)
        └── Resamplea daily a W-FRI (último precio, suma volumen) con ffill entre semanas

─────────────────────────────────────────────────────────────

VariablesTransformation.py
│
└── FeatureEngineer(criterio, ticker_indice)
    ├── feature_cols → lista de columnas features (se rellena al llamar a build())
    ├── build(df_weekly, df_daily) → DataFrame con features + Target, sin NaNs
    │ ├── 1. Variables en DIARIO → resampleadas al último valor de cada viernes
    │ │ ├── RSI_14
    │ │ ├── MACD
    │ │ └── SMA/EMA 50 y 200: log(precio / media)
    │ ├── 2. Variables en SEMANAL
    │ │ ├── Retorno_1W
    │ │ ├── Momentum: 12M, 6M, 1M, Mom_12m_ex_1m (12M - 1M)
    │ │ ├── Momentum relativo vs índice a 3M (RetRel_SPY_3m)
    │ │ ├── Volatilidad: 6M, 1M, ratio Vol_1m/Vol_6m
    │ │ ├── Drawdown 6M (DD_6m)
    │ │ ├── Liquidez: VolumenUSD_z_20, Turnover_20w
    │ │ ├── Lags de retorno: Retorno_t1, Retorno_t2
    │ │ ├── Ratio_52w_High: precio / máximo de 52 semanas
    │ │ ├── Beta_12M vs índice (rolling)
    │ │ └── Bollinger: log(precio/banda_upper), log(precio/banda_lower)
    │ ├── 3. Target
    │ │ ├── criterio='mediana' → 1 si retorno_siguiente > mediana del universo esa semana
    │ │ └── criterio=N (int) → 1 si está entre los N mejores retornos esa semana
    │ └── 4. Imputación: mediana por ticker → fillna(0) → elimina filas sin Target
    └── _build_daily_features(df_daily)
        └── Calcula indicadores diarios y resamplea al último valor semanal (W-FRI)

─────────────────────────────────────────────────────────────

Modelos.py
│
├── ModeloBase (ABC)
│ ├── train(df, feature_cols)
│ │ └── GridSearchCV con WalkForwardCV; guarda el mejor estimador en self.clf
│ └── predict_proba(X) → pd.Series con probabilidad de clase positiva
│
├── RandomForestModel(random_state=42, n_splits=5)
│ └── param_grid: n_estimators, max_depth, min_samples_leaf, max_features, class_weight
│
├── XGBoostModel(random_state=42, n_splits=5)
│ └── param_grid: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, scale_pos_weight
│
└── WalkForwardCV(n_splits=3, val_ratio=0.20)
    └── Cross-validator temporal: entrena desde el inicio, valida en la ventana posterior
        └── Implementa BaseCrossValidator de sklearn (split, get_n_splits, _iter_test_indices)

─────────────────────────────────────────────────────────────

Estrategia.py
│
├── EstrategiaBase (ABC)
│ ├── train(df, feature_cols, tickers_validos, df_daily) → bool
│ │ └── Entrena el modelo sobre tickers válidos; devuelve False si no hay datos
│ └── seleccionar(df_hoy, feature_cols, cartera, df_daily) → dict{ticker: peso}
│
├── EstrategiaMLEquiponderada(modelo, n_activos_obj, umbral_salida)
│ └── seleccionar()
│     ├── Puntúa activos con modelo.predict_proba()
│     ├── Buffer de permanencia: mantiene activos ya en cartera si siguen en el top umbral_salida
│     └── Pesos equiponderados: 1/n_activos entre los seleccionados
│
├── EstrategiaMLMonteCarlo(modelo, n_activos_obj, umbral_salida, peso_max, n_simulaciones, peso_min, dias_retorno)
│ └── seleccionar()
│     ├── Preselecciona candidatos igual que EstrategiaMLEquiponderada
│     └── Optimiza pesos por simulación de Montecarlo maximizando Sharpe con retornos diarios históricos
│
├── EstrategiaMLEquiponderadaMacro(modelo, n_activos_obj, umbral_salida, ticker_indice, umbral_vol, exposicion_rv)
│ ├── Hereda de EstrategiaMLEquiponderada
│ ├── _señal_riesgo(fecha_hoy, df_daily) → bool
│ │ └── True si la vol realizada (20d) del índice supera umbral_vol
│ └── seleccionar()
│     └── Reduce exposición a RV al exposicion_rv% cuando la señal de riesgo se activa
│
└── EstrategiaMLMinVarAlphaTilt(modelo, n_activos_obj, umbral_salida, ...)
    ├── train() → estima covarianza robusta (Ledoit-Wolf) sobre retornos semanales de train
    └── seleccionar()
        ├── Alpha = score ML centrado en p_neutral (clipeado a 0 para long-only)
        ├── Optimización SLSQP: minimiza -(alpha·w) + lambda_risk·(w·Σ·w) + lambda_tc·|Δw|
        ├── No-trade band por activo (no_trade_band)
        ├── Cap de turnover total (turnover_max)
        └── Gate de mejora neta: no opera si mejora < coste_estimado + utility_buffer

─────────────────────────────────────────────────────────────

Backtest.py
│
└── BacktestEngine(universo, proveedor, feature_engineer, estrategia,
                   start_date, end_date, len_ventana, nominal)
    ├── Al instanciar: descarga df_daily y df_weekly completos y construye el df de features
    ├── _train(fecha_pivote, tickers_validos) → bool
    │ └── Entrena la estrategia con datos desde fecha_pivote - len_ventana años
    ├── _ajustar_pesos(cartera, precios_hoy) → dict{ticker: peso_real}
    │ └── Calcula los pesos actuales de la cartera a precios de hoy
    ├── _ajustar_cartera(cartera, datos_hoy, pesos_nuevos) → (cartera, pesos_adj, VP)
    │ └── Compra/vende/ajusta posiciones para alcanzar pesos_nuevos, aplicando costes
    ├── _run() → DataFrame [Fecha, Valor cartera]
    │ ├── Loop diario con mark-to-market; reajuste semanal (viernes)
    │ ├── Re-entrena cada 6 meses
    │ └── Llama a estrategia.seleccionar() y _ajustar_cartera() en fechas semanales
    ├── print_results(bmks, bmk_equal_weight, plot, oracle) → (fechas, serie, metrics_view)
    │ ├── Llama a _run() internamente
    │ ├── Compara con benchmarks y/o benchmark equiponderado
    │ └── Imprime tabla de métricas con build_metrics_table()
    └── _serie_oraculo(n=15) → pd.Series
        └── Benchmark teórico: selecciona siempre los n mejores retornos de la semana siguiente

─────────────────────────────────────────────────────────────

MotorInversion.py
│
└── MotorInversion(universo, feature_engineer, estrategia, estado_path,
                   len_ventana, capital_total, proveedor_cls)
    ├── Al instanciar: carga cartera, fecha de último entrenamiento y modelo desde disco
    ├── ejecutar(fecha) → DataFrame de señales
    │ └── Reentrena si toca → genera señales → guarda cartera, historial y modelo
    ├── _reentrenar_si_toca(fecha)
    │ └── Reentrena cada MESES_RETRAIN (6) meses con datos hasta fecha - 2 semanas
    ├── _descargar_datos(fecha, long_hist) → (df_daily, df_weekly)
    │ └── Descarga desde (fecha - long_hist - 1) años hasta fecha+1 día
    ├── _generar_señales(fecha) → DataFrame [Ticker, Accion, Cantidad, Precio, CT, Precio_Ejecutado]
    │ ├── Llama a estrategia.seleccionar() para obtener pesos objetivo
    │ └── Genera órdenes de COMPRA / VENTA / MANTENER actualizando self.cartera
    ├── _cargar_cartera() / _guardar_cartera()
    │ └── Persiste la cartera en cartera_actual.json
    ├── _guardar_historial(fecha, señales)
    │ └── Añade las operaciones al CSV historial_operaciones.csv (append)
    ├── _leer_fecha_train() / (escritura en _reentrenar_si_toca)
    │ └── Persiste la fecha en ultimo_entrenamiento.txt
    └── _cargar_modelo() / _guardar_modelo()
        └── Serializa/deserializa self.estrategia.modelo con pickle

─────────────────────────────────────────────────────────────

FLUJO DE DATOS
│
│  tickers + csv_cambios
│       ↓
│  UniversoActivos  ──────────────────────────────────────────────────────┐
│       ↓                                                                 ↓
│  tickers válidos                                              BacktestEngine / MotorInversion
│       ↓                                                                 ↑
│  YFinanceProvider                                                       │
│  (df_daily, df_weekly)                                                  │
│       ↓                                                                 │
│  FeatureEngineer                                                        │
│  (df con features + Target)                                             │
│       ↓                                                                 │
│  Estrategia                                                             │
│  (train → seleccionar → pesos)  ────────────────────────────────────────┘

─────────────────────────────────────────────────────────────

USO (Backtest)
│
│  universo   = UniversoActivosDinamico(tickers_hoy, start_date, end_date, csv_path)
│  proveedor  = YFinanceProvider()
│  fe         = FeatureEngineer(criterio=5, ticker_indice="^STOXX50E")
│  modelo     = RandomForestModel()
│  estrategia = EstrategiaMLEquiponderada(modelo, n_activos_obj=15, umbral_salida=22)
│
│  engine = BacktestEngine(universo, proveedor, fe, estrategia,
│                          start_date, end_date, len_ventana=4, nominal=1_000_000)
│  engine.print_results(bmks=["^STOXX50E"])

USO (Motor en producción, ejecución semanal)
│
│  universo   = UniversoActivosEstatico(tickers)
│  fe         = FeatureEngineer(criterio=5, ticker_indice="^STOXX50E")
│  estrategia = EstrategiaMLEquiponderada(modelo, n_activos_obj=15, umbral_salida=22)
│
│  motor = MotorInversion(universo, fe, estrategia, estado_path="./mi_cartera",
│                         len_ventana=4, capital_total=10_000_000,
│                         proveedor_cls=YFinanceProvider)
│  señales = motor.ejecutar(date(2026, 3, 20))