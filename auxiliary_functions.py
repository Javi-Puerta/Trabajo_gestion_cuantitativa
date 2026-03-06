import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    '''
    Calcula el RSI (Relative Strength Index) para una serie de precios.
    El RSI es un indicador de momentum que mide la velocidad y el cambio de los movimientos de precios.
    Este indicador nos indica cuando un activo ha subido demasiado, para no comprarlo.
    '''
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    '''
    Calcula el MACD (Moving Average Convergence Divergence) para una serie de precios.
    El MACD es un indicador de momentum que muestra la relación entre dos medias móviles de los
    precios. Nos indica cuándo un activo comienza a subir, para comprarlo.
    '''
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    return exp1 - exp2

def calculate_bollinger(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """
    Calcula las bandas de Bollinger superior e inferior.
    Devuelve (upper_band, lower_band).

    Parámetros
    ----------
    prices  : pd.Series — precios de cierre diarios
    period  : int       — ventana de la media móvil (default 20)
    num_std : float     — número de desviaciones típicas (default 2)
    """
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()

    upper_band = sma + num_std * std
    lower_band = sma - num_std * std

    return upper_band, lower_band

def calculate_beta(returns: pd.Series, market_returns: pd.Series, period: int) -> pd.Series:
    """
    Calcula la beta rolling de un activo respecto al mercado.

    Parámetros
    ----------
    returns        : pd.Series — retornos semanales del activo
    market_returns : pd.Series — retornos semanales del índice de referencia
    period         : int       — ventana en semanas (ej: 52 para 12M)
    """
    aligned = market_returns.reindex(returns.index)
    cov = returns.rolling(period).cov(aligned)
    var = aligned.rolling(period).var()
    return cov / var.replace(0, np.nan)