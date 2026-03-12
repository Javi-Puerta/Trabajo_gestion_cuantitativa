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
└── Backtest.py

─────────────────────────────────────────────────────────────

auxiliary_functions.py
│
├── calculate_rsi(prices, period)
│ └── RSI con media exponencial (Wilder)
│
├── calculate_macd(prices, fast, slow, signal)
│ └── Diferencia entre EMA rápida y lenta
│
├── calculate_bollinger(prices, period, num_std)
│ └── Devuelve (upper_band, lower_band)
│
└── calculate_beta(returns, market_returns, period)
└── Beta rolling: cov(activo, mercado) / var(mercado)
└── compute_performance_metrics(series, periods_per_year)
└── Calcula retorno/vol anualizados y Sharpe (ya está anualizado)

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
├── _load_changes() → carga y limpia CSV de cambios históricos
├── get_full_ticker_list() → tickers actuales + históricos
└── get_universe_at_date(date) → reconstruye universo en fecha dada

─────────────────────────────────────────────────────────────

ProveedorDatos.py
│
├── ProveedorDatosBase (ABC)
│ ├── download_daily_data() → DataFrame diario
│ └── download_prices_weekly() → DataFrame semanal
│
└── YFinanceProvider(tickers, start_date, end_date)
├── df_daily → ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD']
├── df_weekly → ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD']
├── _download_daily_data() → precios diarios + Volumen_USD = Precio × Volumen
└── _download_prices_weekly() → resamplea df_daily a W-WED (último precio, suma volumen)
└── Nota: llenar faltantes con ffill/bfill para evitar NaNs en backtest

─────────────────────────────────────────────────────────────

VariablesTransformation.py
│
└── FeatureEngineer(criterio, ticker_indice)
├── feature_cols → lista de columnas que usará el modelo (rellenada en build())
├── build(df_weekly, df_daily) → DataFrame features + Target, sin NaNs
│ ├── 1. Variables en DIARIO → resampleadas a semanal
│ │ ├── RSI 14D, 9D, 3D
│ │ ├── MACD
│ │ ├── Bollinger: log(precio/banda_upper), log(precio/banda_lower)
│ │ ├── SMA: log(precio/SMA_200,100,50)
│ │ └── EMA: log(precio/EMA_200,100,50)
│ ├── 2. Variables en SEMANAL
│ │ ├── Retorno_1W
│ │ ├── Momentum 12M, 6M, 1M
│ │ ├── Momentum relativo vs índice (12M)
│ │ ├── Volatilidad 12M, 6M, 1M
│ │ ├── Beta 12M vs índice
│ │ └── Lagged returns: Retorno_t1, Retorno_t2
│ ├── 3. Target
│ │ ├── 'mediana' → 1 si retorno > mediana del universo esa semana
│ │ └── Top-N → 1 si está entre los N mejores retornos esa semana
│ └── 4. dropna() → elimina filas con NaN
└── _build_daily_features(df_daily)
└── Calcula indicadores diarios y resamplea al último valor semanal

─────────────────────────────────────────────────────────────

Modelos.py
│
├── ModeloBase (ABC)
│ ├── train(X, y)
│ └── predict_proba(X) → pd.Series con probabilidad de clase positiva
│
└── RandomForestModel(n_estimators, max_depth, class_weight, random_state)
├── train(X, y)
└── predict_proba(X)
└── XGBoostModel wrapper disponible (misma interfaz)

─────────────────────────────────────────────────────────────

Estrategia.py
│
├── EstrategiaBase (ABC)
│ ├── train(df, feature_cols, tickers_validos)
│ └── seleccionar(df_hoy, feature_cols) → dict{ticker: peso}
│
└── EstrategiaMLEquiponderada(modelo, n_activos_obj, umbral_salida)
├── _cartera_actual → buffer de permanencia entre semanas
├── train(df, feature_cols, tickers_validos)
│ └── Delega en modelo.train()
└── seleccionar(df_hoy, feature_cols) → dict{ticker: peso}
├── Scoring con model.predict_proba()
├── Gestión de delists / tickers no válidos
├── Selección con buffer (umbral_salida)
└── Pesos equiponderados: 1/n_activos_obj por ticker

└── EstrategiaMLMinVarAlphaTilt (optimización)
└── Usa alpha del ML + Ledoit-Wolf + SLSQP (lambda_risk, lambda_tc, w_max, turnover)

─────────────────────────────────────────────────────────────

Backtest.py
│
└── BacktestEngine(universo, proveedor, feature_engineer, estrategia,
start_date, end_date, len_ventana)
├── df → dataset completo construido al instanciar (fe.build())
├── composiciones → {fecha: set(tickers)} precalculado en run()
├── run(coste_operacion)
│ ├── Precalcula composiciones para cada fecha
│ ├── Loop semanal:
│ │ ├── Re-entrena cada 6 meses (estrategia.train)
│ │ ├── estrategia.seleccionar() → dict{ticker: peso}
│ │ └── Calcula retorno como suma ponderada + costes
│ └── Devuelve (DataFrame resultados, rendimiento_total %)
├── _mark_to_market / pesos_adj → cálculo M2M diario
└── print_results(plot=True/False)
└── Ahora devuelve (fechas, serie_estrategia, metrics_view); plot=False suprime gráficos

─────────────────────────────────────────────────────────────

FLUJO DE DATOS
│
│  tickers + csv_cambios
│       ↓
│  UniversoActivos  ──────────────────────────────→  BacktestEngine
│       ↓                                                   ↑
│  tickers válidos                                          │
│       ↓                                                   │
│  YFinanceProvider                                         │
│  (df_daily, df_weekly)                                    │
│       ↓                                                   │
│  FeatureEngineer                                          │
│  (df con features + Target)                               │
│       ↓                                                   │
│  Estrategia                                               │
│  (train → seleccionar → pesos)  ──────────────────────────┘

─────────────────────────────────────────────────────────────

USO
│
│  # Universo estático (fondos, ETFs)
│  universo   = UniversoActivosEstatico(tickers)
│
│  # Universo dinámico (índices con cambios históricos)
│  universo   = UniversoActivosDinamico(tickers_hoy, start_date, end_date, csv_path)
│
│  # Proveedor — incluir siempre el ticker del índice (SPY) para beta y momentum relativo
│  proveedor  = YFinanceProvider(tickers + ["SPY"], start_date, end_date)
│
│  fe         = FeatureEngineer(criterio="mediana", ticker_indice="SPY")
│  modelo     = RandomForestModel()
│  estrategia = EstrategiaMLEquiponderada(modelo, n_activos_obj=5, umbral_salida=7)
│
│  engine     = BacktestEngine(universo, proveedor, fe, estrategia,
│                              start_date, end_date, len_ventana)
│
│  resultados, total = engine.run()