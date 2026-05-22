import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


TEMA_PRESENTACION = {
    "fondo": "#000A26", "panel": "#06133A", "header": "#159A9C",
    "lineas": "#27456F", "blanco": "#FFFFFF",
    "positivo": "#2FE6D0", "negativo": "#FF6B6B",
}


def pesos_ewma(n, lambda_=0.94):
    w = (1 - lambda_) * lambda_ ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def media_ewma(retornos, lambda_=0.94):
    retornos = retornos.dropna(axis=1, how="all").fillna(0.0)
    w = pesos_ewma(len(retornos), lambda_)
    return pd.Series(w @ retornos.values, index=retornos.columns)


def cov_ewma(retornos, lambda_=0.94, centrar=True):
    retornos = retornos.dropna(axis=1, how="all").fillna(0.0)

    if len(retornos) < 2:
        return pd.DataFrame()

    w = pesos_ewma(len(retornos), lambda_)
    x = retornos.values

    if centrar:
        mu = w @ x
        x = x - mu

    cov = x.T @ (x * w[:, None])

    return pd.DataFrame(cov, index=retornos.columns, columns=retornos.columns)


def riesgo_normal_posicion(
    pesos,
    retornos_hist,
    nav,
    nivel=0.99,
    horizonte=20,
    lambda_=0.94,
    usar_media=True,
):
    pesos = pesos[pesos.abs() > 1e-12].dropna()
    tickers = pesos.index.intersection(retornos_hist.columns)

    cols_nan = {
        "VaR %": np.nan, "VaR €": np.nan,
        "TailVaR %": np.nan, "TailVaR €": np.nan,
        "EaR %": np.nan, "EaR €": np.nan,
        "EaR/VaR": np.nan,
        "Volatilidad 1d": np.nan,
        "Volatilidad horizonte": np.nan,
        "Retorno esperado horizonte": np.nan,
    }

    if len(tickers) < 2 or len(retornos_hist) < 20:
        return cols_nan

    ret = retornos_hist[tickers]
    w = pesos[tickers].values

    mu_1d = media_ewma(ret, lambda_=lambda_).values if usar_media else np.zeros(len(tickers))
    cov = cov_ewma(ret, lambda_=lambda_, centrar=usar_media).values

    mu_p_1d = float(w @ mu_1d)
    vol_1d = float(np.sqrt(w @ cov @ w))

    mu_h = mu_p_1d * horizonte
    vol_h = vol_1d * np.sqrt(horizonte)

    alpha = 1 - nivel
    z_inf = norm.ppf(alpha)
    z_sup = norm.ppf(nivel)

    q_inf = mu_h + z_inf * vol_h
    q_sup = mu_h + z_sup * vol_h

    var_pct = max(-q_inf, 0.0)
    ear_pct = max(q_sup, 0.0)

    tail_return = mu_h - vol_h * norm.pdf(z_inf) / alpha
    tailvar_pct = max(-tail_return, 0.0)

    ratio = ear_pct / var_pct if var_pct > 0 else np.nan

    return {
        "VaR %": float(var_pct),
        "VaR €": float(nav * var_pct),
        "TailVaR %": float(tailvar_pct),
        "TailVaR €": float(nav * tailvar_pct),
        "EaR %": float(ear_pct),
        "EaR €": float(nav * ear_pct),
        "EaR/VaR": float(ratio),
        "Volatilidad 1d": float(vol_1d),
        "Volatilidad horizonte": float(vol_h),
        "Retorno esperado horizonte": float(mu_h),
    }


def riesgo_benchmark_actual(
    pesos_bmk,
    retornos_activos,
    fecha,
    nav,
    ventana=550,
    nivel=0.99,
    horizonte=20,
    lambda_=0.94,
    min_obs=250,
    usar_media=True,
):
    hist = retornos_activos.loc[retornos_activos.index < fecha].tail(ventana)

    if len(hist) < min_obs:
        raise ValueError(
            f"No hay suficiente histórico para el benchmark: {len(hist)} observaciones."
        )

    riesgo = riesgo_normal_posicion(
        pesos=pesos_bmk,
        retornos_hist=hist,
        nav=nav,
        nivel=nivel,
        horizonte=horizonte,
        lambda_=lambda_,
        usar_media=usar_media,
    )

    return pd.Series(riesgo, name="Benchmark")


def calcular_riesgo_diario_posiciones(
    pesos,
    retornos_activos,
    nav,
    ventana=550,
    nivel=0.99,
    horizonte=20,
    lambda_=0.94,
    min_obs=100,
    usar_media=True,
):
    filas = []
    fechas = pesos.index.intersection(nav.index)

    for fecha in fechas:
        pesos_fecha = pesos.loc[fecha]

        if pesos_fecha.abs().sum() <= 1e-12:
            continue

        hist = retornos_activos.loc[retornos_activos.index < fecha].tail(ventana)

        if len(hist) < min_obs:
            continue

        fila = {
            "Fecha": fecha,
            "NAV": nav.loc[fecha],
            "Peso invertido": pesos_fecha.abs().sum(),
        }

        fila.update(
            riesgo_normal_posicion(
                pesos=pesos_fecha,
                retornos_hist=hist,
                nav=nav.loc[fecha],
                nivel=nivel,
                horizonte=horizonte,
                lambda_=lambda_,
                usar_media=usar_media,
            )
        )

        filas.append(fila)

    if not filas:
        return pd.DataFrame()

    return pd.DataFrame(filas).set_index("Fecha")


def resumen_riesgo(df_riesgo, vol_cartera=None, riesgo_bmk=None, vol_bmk=None):
    cols = [
        "VaR %",
        "TailVaR %",
        "EaR %",
        "EaR/VaR",
    ]

    cols = [c for c in cols if c in df_riesgo.columns]

    resumen = pd.DataFrame({
        "Actual": df_riesgo[cols].iloc[-1],
        "Medio": df_riesgo[cols].mean(),
    })

    if riesgo_bmk is not None:
        resumen["Benchmark"] = riesgo_bmk.reindex(cols)

    resumen.loc["Volatilidad anualizada", "Actual"] = vol_cartera
    resumen.loc["Volatilidad anualizada", "Medio"] = np.nan

    if riesgo_bmk is not None:
        resumen.loc["Volatilidad anualizada", "Benchmark"] = vol_bmk

    return resumen


def vol_anualizada_periodo(retornos, periods_per_year=252):
    r = retornos.dropna()
    return np.nan if len(r) < 2 else float(r.std(ddof=1) * np.sqrt(periods_per_year))


def _signo_metrica_riesgo(idx):
    idx = str(idx)
    if idx.startswith(("VaR", "TailVaR")):
        return -1
    if idx.startswith("EaR") and idx != "EaR/VaR":
        return 1
    return 0


def formatear_resumen_riesgo(resumen):
    out = resumen.copy().astype(object)
    for idx in out.index:
        signo = _signo_metrica_riesgo(idx)
        for col in out.columns:
            v = resumen.loc[idx, col]
            if pd.isna(v):
                out.loc[idx, col] = ""
                continue

            x = signo * abs(float(v)) if signo else float(v)

            if str(idx).endswith("%") or str(idx).startswith(("Volatilidad", "Retorno esperado")):
                out.loc[idx, col] = f"{x:+.2%}" if signo else f"{x:.2%}"
            elif str(idx).endswith("€"):
                out.loc[idx, col] = f"{x / 1000:+,.1f} k€" if signo else f"{x / 1000:,.1f} k€"
            elif idx == "EaR/VaR":
                out.loc[idx, col] = f"{float(v):.2f}x"
            else:
                out.loc[idx, col] = f"{x:+.2f}" if signo else f"{x:.2f}"

    return out


def tabla_mpl_presentacion(df, titulo=None, figsize=(7.2, 3.0), dpi=220, bbox=(0.04, 0.07, 0.92, 0.72),
                           anchos=None, columnas_signo=(), tema=None, fontsize=9.5):
    tema = {**TEMA_PRESENTACION, **(tema or {})}
    df = df.copy().astype(str)
    columnas_signo = set(columnas_signo)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(tema["fondo"])
    ax.set_facecolor(tema["fondo"])
    ax.axis("off")

    tabla = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", colLoc="center", bbox=bbox)
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(fontsize)
    tabla.scale(0.92, 1.45)

    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor(tema["lineas"])
        celda.set_linewidth(0.7)
        if anchos and col in anchos:
            celda.set_width(anchos[col])
        if fila == 0:
            celda.set_facecolor(tema["header"])
            celda.get_text().set_color(tema["blanco"])
            celda.get_text().set_weight("bold")
        else:
            celda.set_facecolor(tema["panel"])
            celda.get_text().set_weight("bold")
            valor = str(df.iloc[fila - 1, col])
            if df.columns[col] in columnas_signo and valor.startswith("+"):
                color = tema["positivo"]
            elif df.columns[col] in columnas_signo and valor.startswith("-"):
                color = tema["negativo"]
            else:
                color = tema["blanco"]
            celda.get_text().set_color(color)

    if titulo:
        ax.text(0.5, 0.88, titulo, ha="center", va="bottom", color=tema["blanco"],
                fontsize=14, fontweight="bold", transform=ax.transAxes)
    plt.tight_layout(pad=0.25)
    plt.show()
    return fig, ax


def tabla_resumen_riesgo_presentacion(resumen, titulo="Métricas de riesgo"):
    df = formatear_resumen_riesgo(resumen).reset_index().rename(columns={"index": "Métrica"})
    df["Métrica"] = df["Métrica"].replace({
        "VaR %": "VaR", "TailVaR %": "TailVaR", "EaR %": "EaR",
        "Volatilidad anualizada": "Vol. anualizada",
    })
    anchos = {0: 0.34, **{i: 0.22 for i in range(1, df.shape[1])}}
    return tabla_mpl_presentacion(df, titulo=titulo, figsize=(7.4, 3.0), bbox=(0.04, 0.07, 0.92, 0.72),
                                  anchos=anchos, columnas_signo=df.columns[1:], fontsize=9.6)