import pandas as pd
import numpy as np


from auxiliary_functions import calculate_rsi, calculate_macd, calculate_beta, calculate_bollinger, _rolling_std

class FeatureEngineer:
    """
    Calcular las variables independientes a partir de los precios para entrenar el modelo y calcular
    la variable dependiente (target).

    Parámetros
    ----------
    criterio : int | str
        - Entero N   → target = estar en el Top-N de retorno la semana siguiente
        - 'mediana'  → target = superar la mediana del universo esa semana
    """

    def __init__(self, criterio: int | str, ticker_indice: str ):
        self.criterio = criterio
        self.ticker_indice = ticker_indice
        self.feature_cols: list[str] = []

    def _build_daily_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        df = df_daily.copy().sort_values(["Ticker", "Fecha"])
        df["Fecha"] = pd.to_datetime(df["Fecha"])

        # RSI (solo 14) 
        df["RSI_14"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: calculate_rsi(x, 14))

        # MACD
        df["MACD"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: calculate_macd(x))

        # SMA 50 y 200 (log ratios)
        df["Log_Precio_SMA_50"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: np.log(x / x.rolling(50, min_periods=10).mean()))
        df["Log_Precio_SMA_200"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: np.log(x / x.rolling(200, min_periods=50).mean()))

        # EMA 50 y 200
        df["Log_Precio_EMA_50"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: np.log(x / x.ewm(span=50, adjust=False).mean()))
        df["Log_Precio_EMA_200"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: np.log(x / x.ewm(span=200, adjust=False).mean()))

        # Resamplear al último valor de cada semana
        daily_feature_cols = [
            "RSI_14", "MACD",
            "Log_Precio_SMA_50", "Log_Precio_SMA_200",
            "Log_Precio_EMA_50", "Log_Precio_EMA_200"
        ]

        weekly = (
            df.groupby("Ticker")[["Fecha"] + daily_feature_cols]
            .apply(lambda g: g.set_index("Fecha").resample("W-FRI").last())
            .reset_index()
        )

        return weekly

    def build(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
        df = df_weekly.copy().sort_values(["Ticker", "Fecha"])
        df["Fecha"] = pd.to_datetime(df["Fecha"])

        # 1) features diarios resampleados (sólo los necesarios)
        features_diarias = self._build_daily_features(df_daily)
        df = df.merge(features_diarias, on=["Fecha", "Ticker"], how="left")

        if "Dividendos" not in df.columns:
            df["Dividendos"] = 0.0

        precio_prev = df.groupby("Ticker")["Precio_Close"].shift(1)
        
        # 2) variables semanales
        df["Retorno_1W"] = (df["Precio_Close"] + df["Dividendos"]) / precio_prev - 1

        # Momentum (calculamos 12M para derivar Mom_12m_ex_1m pero no la incluimos en features)
        df["Momentum_12M"] = df.groupby("Ticker")["Precio_Close"].pct_change(52)
        df["Momentum_6M"] = df.groupby("Ticker")["Precio_Close"].pct_change(26)
        df["Momentum_1M"] = df.groupby("Ticker")["Precio_Close"].pct_change(4)
        df["Mom_12m_ex_1m"] = df["Momentum_12M"] - df["Momentum_1M"]

        # Momentum relativo vs índice (fallback neutro)
        ret_indice = (
            df[df["Ticker"] == self.ticker_indice]
            .set_index("Fecha")["Retorno_1W"]
            .sort_index()
        )
        if ret_indice.empty:
            df["RetRel_SPY_3m"] = 0.0
        else:
            ret_3m_ind = ret_indice.rolling(13, min_periods=7).sum()
            df["Ret_3m_activo"] = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: x.rolling(13, min_periods=7).sum())
            df["Ret_3m_ind"] = df["Fecha"].map(ret_3m_ind)
            df["RetRel_SPY_3m"] = df["Ret_3m_activo"] - df["Ret_3m_ind"]

        # Volatilidades (calculadas, pero incluimos solo la corta + ratio)
        df["Volatilidad_6M"] = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: _rolling_std(x, 26))
        df["Volatilidad_1M"] = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: _rolling_std(x, 4))
        df["Vol_ratio_1m_6m"] = df["Volatilidad_1M"] / df["Volatilidad_6M"].replace(0, np.nan)

        # Drawdown 6m
        roll_max_6m = df.groupby("Ticker")["Precio_Close"].transform(lambda x: x.rolling(26, min_periods=10).max())
        df["DD_6m"] = df["Precio_Close"] / roll_max_6m - 1.0

        # Liquidez
        vol_mean_20 = df.groupby("Ticker")["Volumen_USD"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        vol_std_20 = df.groupby("Ticker")["Volumen_USD"].transform(lambda x: x.rolling(20, min_periods=5).std())
        df["VolumenUSD_z_20"] = (df["Volumen_USD"] - vol_mean_20) / vol_std_20.replace(0, np.nan)
        df["Turnover_20w"] = df["Volumen_USD"] / vol_mean_20.replace(0, np.nan)

        # Lags
        df["Retorno_t1"] = df.groupby("Ticker")["Retorno_1W"].shift(1)
        df["Retorno_t2"] = df.groupby("Ticker")["Retorno_1W"].shift(2)

        
        # Ratio 52w High
        roll_max_52w = df.groupby("Ticker")["Precio_Close"].transform(lambda x: x.rolling(52, min_periods=26).max())
        df["Ratio_52w_High"] = df["Precio_Close"] / roll_max_52w

        # Beta 12M vs índice
        ret_indice_series = df[df["Ticker"] == self.ticker_indice].set_index("Fecha")["Retorno_1W"]
        df["Retorno_1W_indice"] = df["Fecha"].map(ret_indice_series)
        df["Beta_12M"] = df.groupby("Ticker")["Retorno_1W"].transform(
            lambda x: calculate_beta(x, df.loc[x.index, "Retorno_1W_indice"], period=52)
        )
        df = df.drop(columns=["Retorno_1W_indice"])

        # Bollinger (log precio respecto a cada banda)
        df["Log_Precio_Boll_Upper"] = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / calculate_bollinger(x, period=20)[0]))
        df["Log_Precio_Boll_Lower"] = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / calculate_bollinger(x, period=20)[1]))

        # Target: el retorno la semana siguiente. Luego se convertirá a clasificación según el criterio elegido.
        df["Retorno_Next_Week"] = df.groupby("Ticker")["Retorno_1W"].shift(-1)
        valid = df["Retorno_Next_Week"].notna()

        if self.criterio == "mediana":
            mediana = df.groupby("Fecha")["Retorno_Next_Week"].transform("median")
            df["Target"] = np.where(valid, (df["Retorno_Next_Week"] > mediana).astype(int), np.nan)
        else:
            rank = df.groupby("Fecha")["Retorno_Next_Week"].rank(method="first", ascending=False)
            df["Target"] = np.where(valid, (rank <= self.criterio).astype(int), np.nan)

        # ----------------------------------------------------------------
        # Features finales (conjunto reducido y balanceado)
        # ----------------------------------------------------------------
        self.feature_cols = [
            "Momentum_1M", "Momentum_6M", "Mom_12m_ex_1m",
            "RetRel_SPY_3m",
            "DD_6m", "Volatilidad_1M", "Vol_ratio_1m_6m",
            "VolumenUSD_z_20", "Turnover_20w",
            "Retorno_t1", "Retorno_t2",
            "RSI_14", "MACD",
            "Log_Precio_SMA_50", "Log_Precio_SMA_200",
            "Log_Precio_EMA_50", "Log_Precio_EMA_200",
            "Ratio_52w_High", "Beta_12M",
            "Log_Precio_Boll_Upper", "Log_Precio_Boll_Lower",
        ]

        # 1) Imputación arrastrando el último valor conocido (forward fill) para evitar mirar al futuro
        feat_present = [c for c in self.feature_cols if c in df.columns]
        if feat_present:
            df[feat_present] = df.groupby("Ticker")[feat_present].transform(lambda g: g.ffill())

        # 2) Si quedan NaNs puntuales (por ejemplo al inicio), rellenar con 0 como último recurso
        df[feat_present] = df[feat_present].fillna(0)

        # 3) Eliminar SOLO las filas que no tengan las features calculadas.
        df = df.dropna(subset=self.feature_cols).copy()

        # 4) Limpiar columnas auxiliares que no queremos devolver
        cols_to_drop = ["Retorno_Next_Week", "Ret_3m_activo", "Ret_3m_ind"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        return df.reset_index(drop=True)