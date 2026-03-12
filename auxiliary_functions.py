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
    sma, std = prices.rolling(period).mean(), prices.rolling(period).std()
    return sma + num_std * std, sma - num_std * std

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
    return returns.rolling(period).cov(aligned) / aligned.rolling(period).var().replace(0, np.nan)

def compute_performance_metrics(level_series: pd.Series, periods_per_year: int = 252,
                                rf_annual: float = 0.0) -> dict:
    """
    Calcula métricas de performance a partir de una serie de niveles de cartera.
    """
    s = level_series.dropna()
    r = s.pct_change().dropna()

    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1

    total_return = (1 + r).prod() - 1
    ann_return = (1 + total_return) ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)

    excess = r - rf_period
    sharpe = np.nan if ann_vol == 0 or pd.isna(ann_vol) else (excess.mean() * periods_per_year) / ann_vol

    downside = np.minimum(excess, 0.0)
    downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
    sortino = np.nan if downside_vol == 0 or pd.isna(downside_vol) else (excess.mean() * periods_per_year) / downside_vol

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    calmar = np.nan if max_dd == 0 or pd.isna(max_dd) else ann_return / abs(max_dd)

    return {
        "Rentabilidad total": total_return,
        "Rentabilidad anualizada": ann_return,
        "Volatilidad anualizada": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Win rate": (r > 0).mean(),
        "Mejor periodo": r.max(),
        "Peor periodo": r.min(),
    }


def build_metrics_table(series_dict: dict[str, pd.Series], periods_per_year: int = 252,
                        rf_annual: float = 0.0) -> pd.DataFrame:
    """
    Recibe {'Nombre': serie_de_nivel} y devuelve una tabla comparativa de métricas.
    """
    return pd.DataFrame({name: compute_performance_metrics(serie, periods_per_year, rf_annual) for name, serie in series_dict.items()}).T