import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COLORES_BT = {
    "XGBoost + MC": "#4F82FF",
    "Random Forest + EW": "#2FE6D0",
    "Random Forest + MC": "#22A109",
    "STOXX 50": "#000000",
    "XGBoost + EW + Macro": "#FF5733",
    "Grad. Boost, Monte Carlo": "#4F82FF",
    "Rand. Forest, Pesos Iguales": "#2FE6D0",
}

TEMAS_BT = {
    "oscuro": dict(fondo="#070A2D", texto="white", grid="white", spine="#6D739C",
                   legend="#101545", legend_edge="#4D5AA0", grid_alpha=0.12),
    "blanco": dict(fondo="white", texto="#111827", grid="#9CA3AF", spine="#9CA3AF",
                   legend="white", legend_edge="#D1D5DB", grid_alpha=0.35),
}

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


def _rolling_std(x, window):
    minp = max(1, window // 2)

    return x.rolling(window, min_periods=minp).std()


def _clip_by_quantiles(df, col, low_q=0.01, high_q=0.99):
        low = df.groupby("Ticker")[col].transform(lambda s: s.quantile(low_q))
        high = df.groupby("Ticker")[col].transform(lambda s: s.quantile(high_q))
        df[col] = df[col].clip(lower=low, upper=high)

        return None


def calcular_costes(tickers: list) -> dict:
    """
    Calcula los costes de transacción de comprar/vender cada activo. Estos costes son estáticos
    para cada activo.
    """
    costes = {}
    for ticker in tickers:
        costes[ticker] = 0.0005 # 0.05% de coste por operación, por ejemplo
    return costes


def mark_to_market(cartera: dict, datos_hoy: pd.DataFrame) -> float:
    '''
    Calcula el valor de la cartera hoy, dada la cantidad en cash, la cantidad de acciones poseídas
    de cada ticker y los precios actuales.
    '''
    valor_cartera = cartera["cash"]
    precios_hoy = datos_hoy.set_index("Ticker")["Precio_Close"]

    for ticker, cantidad in cartera.items():
        if ticker == "cash":
            continue
        
        valor_cartera += cantidad * precios_hoy.get(ticker, np.nan)

    return valor_cartera


def serie_backtest(engine, nombre, normalizar=True):
    s = engine._run().set_index("Fecha")["Valor cartera"].sort_index()
    return (s / s.iloc[0] if normalizar else s).rename(nombre)


def serie_benchmark(proveedor, ticker, fechas, nombre):
    fechas = pd.DatetimeIndex(pd.to_datetime(fechas)).tz_localize(None)
    df = proveedor.download_prices_daily([ticker], (fechas.min()-pd.Timedelta(days=10)).strftime("%Y-%m-%d"), (fechas.max()+pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.tz_localize(None)
    s = df[df["Ticker"].eq(ticker)].set_index("Fecha")["Precio_Close"].sort_index().reindex(fechas).ffill().bfill()
    if s.isna().all(): raise ValueError(f"No hay datos para {ticker}")
    return (s / s.dropna().iloc[0]).rename(nombre)


def tabla_metricas_backtest(series, rf_annual=0.0, nombres=None):
    tabla = build_metrics_table({c: series[c] for c in series.columns}, 252, rf_annual)
    if nombres:
        tabla = tabla.rename(index=nombres)

    return tabla[
        ["Rentabilidad total", "Volatilidad anualizada", "Sharpe", "Max Drawdown"]
    ].rename(columns={
        "Rentabilidad total": "Rentabilidad",
        "Volatilidad anualizada": "Volatilidad",
        "Max Drawdown": "Max DD",
    })


def formatear_tabla_backtest(tabla, tema="oscuro", transponer=True):
    if tema is None:
        return tabla

    df = tabla.copy()
    for c in ["Rentabilidad", "Volatilidad", "Max DD"]:
        if c in df:
            df[c] = df[c].map(lambda x: f"{float(x):.2%}")
    if "Sharpe" in df:
        df["Sharpe"] = df["Sharpe"].map(lambda x: f"{float(x):.2f}")

    df = df.T if transponer else df

    return (
        df.style
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#28669A"), ("color", "white"),
                ("font-weight", "bold"), ("text-align", "center"),
                ("border", "1px solid #8AA6C1"), ("padding", "9px"),
            ]},
            {"selector": "td", "props": [
                ("background-color", "#070A2D"), ("color", "white"),
                ("font-weight", "bold"), ("text-align", "center"),
                ("border", "1px solid #2A3A70"), ("padding", "9px"),
            ]},
        ])
    )


def grafico_backtests(series, titulo="Backtest comparativo", tema="oscuro"):
    cfg = TEMAS_BT[tema]
    series = series.copy().sort_index().ffill()
    series = series.dropna(axis=1, how="all").dropna(how="all")
    dd = series / series.cummax() - 1
    normalizado = series.iloc[0].median() < 10

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8.8), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08}
    )
    fig.patch.set_facecolor(cfg["fondo"])

    for ax in (ax1, ax2):
        ax.set_facecolor(cfg["fondo"])
        ax.grid(True, color=cfg["grid"], alpha=cfg["grid_alpha"], linewidth=0.8)
        ax.tick_params(colors=cfg["texto"], labelsize=11)
        for s in ax.spines.values():
            s.set_color(cfg["spine"])

    fallback = ["#4F82FF", "#2FE6D0", "#FFB84D", "#C084FC", "#F87171"]

    for i, col in enumerate(series.columns):
        color = COLORES_BT.get(col, fallback[i % len(fallback)])
        estilo = "--" if ("STOXX" in col.upper() or "BMK" in col.upper()) else "-"
        lw = 2.3 if estilo == "--" else 2.8

        ax1.plot(series.index, series[col], label=col, color=color, linewidth=lw, linestyle=estilo)
        ax1.scatter(series.index[-1], series[col].iloc[-1], color=color, s=28, zorder=3)
        ax1.text(series.index[-1], series[col].iloc[-1], f"  {series[col].iloc[-1]:.2f}x",
                 color=color, fontsize=10, fontweight="bold", va="center")

        ax2.plot(dd.index, dd[col], label=f"{col} ({dd[col].min():.2%})",
                 color=color, linewidth=2.0, linestyle=estilo)

    ax2.fill_between(dd.index, dd.iloc[:, 0], 0, color="#FF3B30", alpha=0.28)
    ax1.axhline(1 if normalizado else series.iloc[0, 0], color=cfg["texto"],
                linestyle="--", linewidth=1, alpha=0.35)
    ax2.axhline(0, color=cfg["texto"], linewidth=1, alpha=0.65)

    ax1.set_title(titulo, color=cfg["texto"], fontsize=22, fontweight="bold", pad=16)
    ax1.set_ylabel("Valor normalizado" if normalizado else "Valor cartera", color=cfg["texto"], fontsize=13)
    ax2.set_ylabel("Drawdown", color=cfg["texto"], fontsize=13)
    ax2.set_xlabel("Fecha", color=cfg["texto"], fontsize=13)

    if normalizado:
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}x"))
    else:
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.2f} M€"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    min_dd = float(dd.min().min())
    ax2.set_ylim(min_dd * 1.15 if min_dd < 0 else -0.01, 0.003)

    for ax, loc, fs in [(ax1, "upper left", 11), (ax2, "lower left", 10)]:
        leg = ax.legend(loc=loc, frameon=True, fontsize=fs)
        leg.get_frame().set_facecolor(cfg["legend"])
        leg.get_frame().set_edgecolor(cfg["legend_edge"])
        leg.get_frame().set_alpha(0.88)
        for t in leg.get_texts():
            t.set_color(cfg["texto"])

    ax1.margins(x=0.04)
    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)

