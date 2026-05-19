"""
var_funciones.py
================
Funciones auxiliares para el cálculo de VaR (Value at Risk) y
TailVaR / CVaR (Conditional Value at Risk / Expected Shortfall).

Incluye métodos paramétrico (normal) e histórico, tanto para el cálculo
presente (a fecha actual) como para series temporales rolling (histórico).

Horizonte por defecto: 5 días de trading (1 semana).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Optional, List


# ---------------------------------------------------------------------------
# Funciones base
# ---------------------------------------------------------------------------

def calcular_var_parametrico(retornos: pd.Series, nivel_confianza: float = 0.95,
                              horizonte: int = 5) -> float:
    """
    VaR paramétrico asumiendo distribución normal.

    Parámetros
    ----------
    retornos : pd.Series
        Serie de retornos diarios (no porcentuales, p.ej. 0.01 = 1 %).
    nivel_confianza : float
        Nivel de confianza (0.95 → 95 %).
    horizonte : int
        Número de días de trading del horizonte (5 = 1 semana).

    Devuelve
    --------
    float
        VaR como pérdida positiva (p.ej. 0.045 = 4.5 %).
    """
    retornos = retornos.dropna()
    if len(retornos) < 5:
        return np.nan

    mu_diaria  = retornos.mean()
    sigma_diaria = retornos.std(ddof=1)

    # Escalado al horizonte
    mu_h    = mu_diaria * horizonte
    sigma_h = sigma_diaria * np.sqrt(horizonte)

    z = stats.norm.ppf(1 - nivel_confianza)
    var = -(mu_h + z * sigma_h)
    return float(max(var, 0.0))


def calcular_tailvar_parametrico(retornos: pd.Series, nivel_confianza: float = 0.95,
                                  horizonte: int = 5) -> float:
    """
    TailVaR / CVaR / Expected Shortfall paramétrico (distribución normal).

    Parámetros
    ----------
    retornos : pd.Series
        Serie de retornos diarios.
    nivel_confianza : float
        Nivel de confianza (0.95 → 95 %).
    horizonte : int
        Número de días de trading del horizonte (5 = 1 semana).

    Devuelve
    --------
    float
        CVaR como pérdida positiva.
    """
    retornos = retornos.dropna()
    if len(retornos) < 5:
        return np.nan

    mu_diaria    = retornos.mean()
    sigma_diaria = retornos.std(ddof=1)

    mu_h    = mu_diaria * horizonte
    sigma_h = sigma_diaria * np.sqrt(horizonte)

    alpha = 1 - nivel_confianza
    z = stats.norm.ppf(alpha)
    phi_z = stats.norm.pdf(z)

    cvar = -(mu_h - sigma_h * phi_z / alpha)
    return float(max(cvar, 0.0))


def calcular_var_historico(retornos: pd.Series, nivel_confianza: float = 0.95,
                            horizonte: int = 5) -> float:
    """
    VaR histórico (simulación histórica) usando retornos acumulados a `horizonte` días.

    Parámetros
    ----------
    retornos : pd.Series
        Serie de retornos diarios.
    nivel_confianza : float
        Nivel de confianza (0.95 → 95 %).
    horizonte : int
        Número de días de trading del horizonte (5 = 1 semana).

    Devuelve
    --------
    float
        VaR histórico como pérdida positiva.
    """
    retornos = retornos.dropna()
    if len(retornos) < horizonte + 1:
        return np.nan

    # Retornos acumulados a `horizonte` días (log-retornos sumados)
    log_ret = np.log1p(retornos)
    ret_h   = log_ret.rolling(horizonte).sum().dropna()
    ret_h   = np.expm1(ret_h)  # volver a retornos simples

    var = -np.percentile(ret_h, (1 - nivel_confianza) * 100)
    return float(max(var, 0.0))


def calcular_tailvar_historico(retornos: pd.Series, nivel_confianza: float = 0.95,
                                horizonte: int = 5) -> float:
    """
    TailVaR / CVaR histórico (media de pérdidas en la cola) usando retornos
    acumulados a `horizonte` días.

    Parámetros
    ----------
    retornos : pd.Series
        Serie de retornos diarios.
    nivel_confianza : float
        Nivel de confianza (0.95 → 95 %).
    horizonte : int
        Número de días de trading del horizonte (5 = 1 semana).

    Devuelve
    --------
    float
        CVaR histórico como pérdida positiva.
    """
    retornos = retornos.dropna()
    if len(retornos) < horizonte + 1:
        return np.nan

    log_ret = np.log1p(retornos)
    ret_h   = log_ret.rolling(horizonte).sum().dropna()
    ret_h   = np.expm1(ret_h)

    umbral = np.percentile(ret_h, (1 - nivel_confianza) * 100)
    cola   = ret_h[ret_h <= umbral]
    if len(cola) == 0:
        return np.nan

    cvar = -cola.mean()
    return float(max(cvar, 0.0))


# ---------------------------------------------------------------------------
# Funciones de alto nivel
# ---------------------------------------------------------------------------

def calcular_var_presente(retornos: pd.Series,
                           niveles: Optional[List[float]] = None,
                           horizonte: int = 5) -> pd.DataFrame:
    """
    Calcula VaR y TailVaR presentes (paramétrico e histórico) para varios
    niveles de confianza.

    Parámetros
    ----------
    retornos : pd.Series
        Serie de retornos diarios de la cartera.
    niveles : list of float, optional
        Niveles de confianza. Por defecto [0.95, 0.99].
    horizonte : int
        Horizonte en días de trading (5 = 1 semana).

    Devuelve
    --------
    pd.DataFrame
        Tabla con columnas [VaR Param, TailVaR Param, VaR Hist, TailVaR Hist]
        e índice = niveles de confianza.
    """
    if niveles is None:
        niveles = [0.95, 0.99]

    filas = []
    for nc in niveles:
        filas.append({
            "Nivel confianza": f"{nc:.0%}",
            "VaR Param":       calcular_var_parametrico(retornos, nc, horizonte),
            "TailVaR Param":   calcular_tailvar_parametrico(retornos, nc, horizonte),
            "VaR Hist":        calcular_var_historico(retornos, nc, horizonte),
            "TailVaR Hist":    calcular_tailvar_historico(retornos, nc, horizonte),
        })

    df = pd.DataFrame(filas).set_index("Nivel confianza")
    return df


def calcular_var_rolling(retornos: pd.Series,
                          ventana: int = 20,
                          nivel_confianza: float = 0.95,
                          horizonte: int = 5) -> pd.DataFrame:
    """
    Serie temporal (rolling) de VaR y TailVaR para ver la evolución histórica.

    Parámetros
    ----------
    retornos : pd.Series
        Serie de retornos diarios de la cartera.
    ventana : int
        Tamaño de la ventana rolling en días de trading.
    nivel_confianza : float
        Nivel de confianza (0.95 → 95 %).
    horizonte : int
        Horizonte en días de trading (5 = 1 semana).

    Devuelve
    --------
    pd.DataFrame
        Columnas: [VaR Param, TailVaR Param, VaR Hist, TailVaR Hist].
    """
    resultados = {
        "VaR Param":     [],
        "TailVaR Param": [],
        "VaR Hist":      [],
        "TailVaR Hist":  [],
    }
    fechas = []

    for i in range(ventana, len(retornos) + 1):
        ventana_ret = retornos.iloc[i - ventana: i]
        fecha = retornos.index[i - 1]

        resultados["VaR Param"].append(
            calcular_var_parametrico(ventana_ret, nivel_confianza, horizonte))
        resultados["TailVaR Param"].append(
            calcular_tailvar_parametrico(ventana_ret, nivel_confianza, horizonte))
        resultados["VaR Hist"].append(
            calcular_var_historico(ventana_ret, nivel_confianza, horizonte))
        resultados["TailVaR Hist"].append(
            calcular_tailvar_historico(ventana_ret, nivel_confianza, horizonte))
        fechas.append(fecha)

    return pd.DataFrame(resultados, index=pd.DatetimeIndex(fechas))


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def grafico_var_historico(df_rolling: pd.DataFrame,
                           nav: pd.Series,
                           nivel_confianza: float = 0.95,
                           horizonte: int = 5,
                           titulo: str = "VaR y TailVaR — Evolución histórica") -> None:
    """
    Gráfico premium con la evolución del NAV, VaR rolling y TailVaR rolling.

    Parámetros
    ----------
    df_rolling : pd.DataFrame
        Resultado de `calcular_var_rolling()`.
    nav : pd.Series
        Serie con el valor de la cartera (para el gráfico superior).
    nivel_confianza : float
        Nivel de confianza usado (para la leyenda).
    horizonte : int
        Horizonte en días (para el título).
    titulo : str
        Título del gráfico.
    """
    fondo = "#070A2D"
    colores = {
        "VaR Param":     "#4F82FF",
        "TailVaR Param": "#2FE6D0",
        "VaR Hist":      "#FFB84D",
        "TailVaR Hist":  "#FF3B30",
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 9), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.6, 2.4], "hspace": 0.06}
    )
    fig.patch.set_facecolor(fondo)

    for ax in (ax1, ax2):
        ax.set_facecolor(fondo)
        ax.grid(True, color="white", alpha=0.10, linewidth=0.7)
        ax.tick_params(colors="white", labelsize=10)
        for sp in ax.spines.values():
            sp.set_color("#3D4280")

    # --- Panel superior: NAV ---
    nav_alineado = nav.reindex(nav.index.union(df_rolling.index)).ffill()
    nav_alineado = nav_alineado.reindex(df_rolling.index)

    ax1.plot(nav_alineado.index, nav_alineado / 1e6,
             color="#4F82FF", linewidth=2.2, label="NAV (M€)")
    ax1.set_ylabel("Valor cartera (M€)", color="white", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f} M€"))
    leg1 = ax1.legend(loc="upper left", frameon=True, fontsize=9)
    leg1.get_frame().set_facecolor("#101545")
    leg1.get_frame().set_edgecolor("#4D5AA0")
    for t in leg1.get_texts():
        t.set_color("white")

    # --- Panel inferior: VaR / TailVaR ---
    semanas = horizonte // 5
    h_str = f"{semanas} semana" if semanas == 1 else f"{semanas} semanas"
    nc_str = f"{nivel_confianza:.0%}"

    for col, color in colores.items():
        if col in df_rolling.columns:
            ax2.plot(df_rolling.index, df_rolling[col] * 100,
                     color=color, linewidth=2.0, label=f"{col} ({nc_str}, {h_str})")

    ax2.set_ylabel("Pérdida estimada (%)", color="white", fontsize=11)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    leg2 = ax2.legend(loc="upper left", frameon=True, fontsize=9, ncol=2)
    leg2.get_frame().set_facecolor("#101545")
    leg2.get_frame().set_edgecolor("#4D5AA0")
    for t in leg2.get_texts():
        t.set_color("white")

    ax1.set_title(titulo, color="white", fontsize=17, fontweight="bold", pad=12)
    ax2.set_xlabel("Fecha", color="white", fontsize=11)

    plt.tight_layout()
    plt.show()


def formatear_tabla_presente(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Da formato visual (Styler) a la tabla de VaR/TailVaR presente.

    Parámetros
    ----------
    df : pd.DataFrame
        Resultado de `calcular_var_presente()`.

    Devuelve
    --------
    pd.io.formats.style.Styler
        Tabla con formato de porcentaje y colores.
    """
    def pct(x):
        return f"{x:.2%}" if pd.notna(x) else "N/A"

    return (
        df.style
        .format(pct)
        .set_caption("VaR y TailVaR presentes")
        .set_table_styles([
            {"selector": "th",
             "props": [("background-color", "#28669A"), ("color", "white"),
                       ("font-weight", "bold"), ("text-align", "center"),
                       ("border", "1px solid #8AA6C1")]},
            {"selector": "td",
             "props": [("background-color", "white"), ("color", "black"),
                       ("font-weight", "bold"), ("text-align", "center"),
                       ("border", "1px solid #E0E0E0"), ("padding", "10px")]},
            {"selector": "caption",
             "props": [("caption-side", "top"), ("font-weight", "bold"),
                       ("font-size", "14px"), ("color", "#1E2A4A")]},
        ])
    )
