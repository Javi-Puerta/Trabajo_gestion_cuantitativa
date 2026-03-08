import pandas as pd
import numpy as np

from auxiliary_functions import calculate_rsi, calculate_macd, calculate_bollinger, calculate_beta

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

    @staticmethod
    def _rolling_std(x, window):
        minp = max(1, window // 2)
        return x.rolling(window, min_periods=minp).std()

    # helper: clip por quantiles por Ticker
    @staticmethod
    def _clip_by_quantiles(df, col, low_q=0.01, high_q=0.99):
        low = df.groupby("Ticker")[col].transform(lambda s: s.quantile(low_q))
        high = df.groupby("Ticker")[col].transform(lambda s: s.quantile(high_q))
        df[col] = df[col].clip(lower=low, upper=high)

    def build(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
        df = df_weekly.copy().sort_values(["Ticker", "Fecha"])
        df["Fecha"] = pd.to_datetime(df["Fecha"])

        # 1) features diarios resampleados (sólo los necesarios)
        features_diarias = self._build_daily_features(df_daily)
        df = df.merge(features_diarias, on=["Fecha", "Ticker"], how="left")

        # 2) variables semanales
        df["Retorno_1W"] = df.groupby("Ticker")["Precio_Close"].pct_change(1)

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
        df["Volatilidad_6M"] = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: self._rolling_std(x, 26))
        df["Volatilidad_1M"] = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: self._rolling_std(x, 4))
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

        # Target (igual que antes)
        df["Retorno_Next_Week"] = df.groupby("Ticker")["Retorno_1W"].shift(-1)
        if self.criterio == "mediana":
            mediana = df.groupby("Fecha")["Retorno_Next_Week"].transform("median")
            df["Target"] = (df["Retorno_Next_Week"] > mediana).astype(int)
        else:
            rank = df.groupby("Fecha")["Retorno_Next_Week"].rank(method="first", ascending=False)
            df["Target"] = (rank <= self.criterio).astype(int)

        # ----------------------------------------------------------------
        # Winsorize / clip de columnas problemáticas (por ticker)
        # ----------------------------------------------------------------
        for c in ["Momentum_1M", "Momentum_6M", "Mom_12m_ex_1m", "Volatilidad_1M", "Vol_ratio_1m_6m", "DD_6m", "VolumenUSD_z_20", "Turnover_20w"]:
            if c in df.columns:
                try:
                    self._clip_by_quantiles(df, c, 0.01, 0.99)
                except Exception:
                    pass

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
            "Log_Precio_EMA_50", "Log_Precio_EMA_200"
        ]

        # 1) Imputación por mediana por ticker para conservar filas
        feat_present = [c for c in self.feature_cols if c in df.columns]
        if feat_present:
            df[feat_present] = df.groupby("Ticker")[feat_present].transform(lambda g: g.fillna(g.median()))

        # 2) Si quedan NaNs puntuales (por ejemplo al inicio), rellenar con 0 como último recurso
        df[feat_present] = df[feat_present].fillna(0)

        # 3) Eliminar filas sin Target (última fila de cada ticker o series incompletas)
        df = df[df["Target"].notna()].copy()

        # 4) Limpiar columnas auxiliares que no queremos devolver
        cols_to_drop = ["Retorno_Next_Week", "Ret_3m_activo", "Ret_3m_ind"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        return df.reset_index(drop=True)

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
            .apply(lambda g: g.set_index("Fecha").resample("W-WED").last())
            .reset_index()
        )

        return weekly