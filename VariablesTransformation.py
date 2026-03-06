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

    def build(self, df_weekly: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe:
            df_weekly : ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD']
            df_daily  : ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD']
        Devuelve el DataFrame semanal enriquecido con features y Target, sin NaNs.
        """
        df = df_weekly.copy().sort_values(["Ticker", "Fecha"])

        # ----------------------------------------------------------------
        # 1. Variables calculadas en DIARIO → resampleadas a semanal
        # ----------------------------------------------------------------
        features_diarias = self._build_daily_features(df_daily)
        df = df.merge(features_diarias, on=["Fecha", "Ticker"], how="left")

        # ----------------------------------------------------------------
        # 2. Variables calculadas en SEMANAL
        # ----------------------------------------------------------------

        # Retornos semanales (necesarios para momentum, vol, beta y target)
        df["Retorno_1W"] = df.groupby("Ticker")["Precio_Close"].pct_change(1)

        # Momentum
        df["Momentum_12M"] = df.groupby("Ticker")["Precio_Close"].pct_change(52)
        df["Momentum_6M"]  = df.groupby("Ticker")["Precio_Close"].pct_change(26)
        df["Momentum_1M"]  = df.groupby("Ticker")["Precio_Close"].pct_change(4)

        # Momentum relativo vs índice
        retorno_indice = df[df["Ticker"] == self.ticker_indice].set_index("Fecha")["Retorno_1W"]
        df["Retorno_Indice"] = df["Fecha"].map(retorno_indice)
        df["Momentum_Relativo_12M"] = (
            df.groupby("Ticker")["Precio_Close"].pct_change(52) -
            df.groupby("Fecha")["Retorno_Indice"].transform("first").rolling(52).sum()  
        )

        # Volatilidad rolling
        df["Volatilidad_12M"] = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: x.rolling(52).std())
        df["Volatilidad_6M"]  = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: x.rolling(26).std())
        df["Volatilidad_1M"]  = df.groupby("Ticker")["Retorno_1W"].transform(lambda x: x.rolling(4).std())

        # Beta 12M vs índice
        retornos_indice_serie = df[df["Ticker"] == self.ticker_indice].set_index("Fecha")["Retorno_1W"]
        df["Beta_12M"] = df.groupby("Ticker", group_keys=False).apply(
            lambda g: calculate_beta(g.set_index("Fecha")["Retorno_1W"], retornos_indice_serie, 52)
        ).values

        # Lagged returns
        df["Retorno_t1"] = df.groupby("Ticker")["Retorno_1W"].shift(1)
        df["Retorno_t2"] = df.groupby("Ticker")["Retorno_1W"].shift(2)

        # ----------------------------------------------------------------
        # 3. Target
        # ----------------------------------------------------------------
        df["Retorno_Next_Week"] = df.groupby("Ticker")["Retorno_1W"].shift(-1)

        if self.criterio == "mediana":
            mediana = df.groupby("Fecha")["Retorno_Next_Week"].transform("median")
            df["Target"] = (df["Retorno_Next_Week"] > mediana).astype(int)
        else:
            rank = df.groupby("Fecha")["Retorno_Next_Week"].rank(method="first", ascending=False)
            df["Target"] = (rank <= self.criterio).astype(int)

        # ----------------------------------------------------------------
        # 4. Feature cols
        # ----------------------------------------------------------------
        self.feature_cols = [
            "Momentum_12M", "Momentum_6M", "Momentum_1M", "Momentum_Relativo_12M",
            "Volatilidad_12M", "Volatilidad_6M", "Volatilidad_1M", "Beta_12M",
            "Retorno_t1", "Retorno_t2", "Volumen_USD",
            "RSI_14", "RSI_9", "RSI_3",
            "Log_Precio_Bollinger_Upper", "Log_Precio_Bollinger_Lower",
            "Log_Precio_SMA_200", "Log_Precio_SMA_100", "Log_Precio_SMA_50",
            "Log_Precio_EMA_200", "Log_Precio_EMA_100", "Log_Precio_EMA_50",
            "MACD"
        ]

        return df.drop(columns=["Retorno_Next_Week", "Precio_Close", "Retorno_Indice"],
                       errors="ignore").dropna()

    def _build_daily_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores en frecuencia diaria y resamplea al último valor de cada semana.
        Devuelve DataFrame con ['Fecha', 'Ticker', ...features_diarias...]
        """
        df = df_daily.copy().sort_values(["Ticker", "Fecha"])

        # RSI
        df["RSI_14"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: calculate_rsi(x, 14))
        df["RSI_9"]  = df.groupby("Ticker")["Precio_Close"].transform(lambda x: calculate_rsi(x, 9))
        df["RSI_3"]  = df.groupby("Ticker")["Precio_Close"].transform(lambda x: calculate_rsi(x, 3))

        # Bollinger
        def add_bollinger(g):
            upper, lower = calculate_bollinger(g["Precio_Close"])
            g["Log_Precio_Bollinger_Upper"] = np.log(g["Precio_Close"] / upper)
            g["Log_Precio_Bollinger_Lower"] = np.log(g["Precio_Close"] / lower)
            return g
        df = df.groupby("Ticker", group_keys=False).apply(add_bollinger)

        # SMA
        df["Log_Precio_SMA_200"] = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / x.rolling(200).mean()))
        df["Log_Precio_SMA_100"] = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / x.rolling(100).mean()))
        df["Log_Precio_SMA_50"]  = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / x.rolling(50).mean()))
        
        # MACD
        df["MACD"] = df.groupby("Ticker")["Precio_Close"].transform(lambda x: calculate_macd(x))

        # EMA
        df["Log_Precio_EMA_200"] = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / x.ewm(span=200, adjust=False).mean()))
        df["Log_Precio_EMA_100"] = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / x.ewm(span=100, adjust=False).mean()))
        df["Log_Precio_EMA_50"]  = df.groupby("Ticker")["Precio_Close"].transform(
            lambda x: np.log(x / x.ewm(span=50, adjust=False).mean()))

        # Resamplear al último valor de cada semana
        daily_feature_cols = [
            "RSI_14", "RSI_9", "RSI_3",
            "Log_Precio_Bollinger_Upper", "Log_Precio_Bollinger_Lower",
            "Log_Precio_SMA_200", "Log_Precio_SMA_100", "Log_Precio_SMA_50",
            "Log_Precio_EMA_200", "Log_Precio_EMA_100", "Log_Precio_EMA_50",
            "MACD"
        ]

        df["Fecha"] = pd.to_datetime(df["Fecha"])
        weekly = (
            df.groupby("Ticker")[["Fecha"] + daily_feature_cols]
            .apply(lambda g: g.set_index("Fecha").resample("W-WED").last())
            .reset_index()
        )

        return weekly