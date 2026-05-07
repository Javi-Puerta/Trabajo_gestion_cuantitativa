import pandas as pd
import yfinance as yf
import unicodedata
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def get_eurostoxx50_tickers():
    url = 'https://en.wikipedia.org/wiki/EURO_STOXX_50'
    headers = {"User-Agent": "Mozilla/5.0 ..."}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text), flavor='bs4')
    df = next(t for t in tables if 'Ticker' in t.columns)
    return df['Ticker'].tolist()


def _limpiar_col(c):
    c = unicodedata.normalize("NFKD", str(c))
    c = "".join(x for x in c if not unicodedata.combining(x))
    return c.lower().strip().replace(" ", "_")


def _leer_operaciones(archivo, fecha_fin=None, hoja="Operativa", incluir_costes=True):
    ops = pd.read_excel(archivo, sheet_name=hoja) if str(archivo).endswith((".xlsx", ".xls")) else pd.read_csv(archivo)

    ops.columns = [_limpiar_col(c) for c in ops.columns]
    ops = ops.rename(columns={"id": "ticker"})

    ops["fecha"] = pd.to_datetime(ops["fecha"], dayfirst=True)
    ops["accion"] = ops["accion"].astype(str).str.upper().str.strip()

    ops = ops[
        ops["accion"].isin(["COMPRA", "VENTA"])
        & ops["ticker"].notna()
    ].copy()

    ops["ticker"] = ops["ticker"].astype(str).str.strip()
    ops = ops[~ops["ticker"].isin(["CASH", "nan", "NaN", ""])].copy()

    fecha_fin = pd.Timestamp.today().normalize() if fecha_fin is None else pd.to_datetime(fecha_fin, dayfirst=True).normalize()
    ops = ops[ops["fecha"] <= fecha_fin].copy()

    if ops.empty:
        raise ValueError(f"No hay compras/ventas hasta {fecha_fin.date()}.")

    if "precio_ejecutado" not in ops:
        ops["precio_ejecutado"] = ops["precio"]

    if not incluir_costes:
        ops["precio_ejecutado"] = ops["precio"]

    for c in ["cantidad", "precio", "precio_ejecutado"]:
        ops[c] = pd.to_numeric(ops[c], errors="coerce")

    return ops.sort_values("fecha").reset_index(drop=True), fecha_fin


def _campo_yf(datos, campo, tickers):
    if isinstance(datos.columns, pd.MultiIndex):
        if campo in datos.columns.get_level_values(0):
            out = datos[campo]
        elif campo in datos.columns.get_level_values(1):
            out = datos.xs(campo, axis=1, level=1)
        else:
            out = pd.DataFrame(0.0, index=datos.index, columns=tickers)
    else:
        out = datos[campo].to_frame(tickers[0]) if campo in datos else pd.DataFrame(0.0, index=datos.index, columns=tickers)

    return out.reindex(columns=tickers)


def _datos_mercado(ops, fecha_fin, universo_tickers=()):
    tickers = sorted(set(ops["ticker"]) | set(universo_tickers))
    start = ops["fecha"].min()
    end = fecha_fin + pd.Timedelta(days=2)

    if start >= end:
        raise ValueError(f"Fechas inválidas: start={start.date()} >= end={end.date()}")

    datos = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        actions=True,
        auto_adjust=False,
        progress=False,
    )

    precios = _campo_yf(datos, "Close", tickers).ffill()
    dividendos = _campo_yf(datos, "Dividends", tickers).reindex(precios.index).fillna(0.0)

    if precios.empty:
        raise ValueError("yfinance no ha devuelto precios.")

    def siguiente_sesion(f):
        validas = precios.index[precios.index >= f]
        return validas[0] if len(validas) else pd.NaT

    ops = ops.copy()
    ops["fecha_trade"] = ops["fecha"].apply(siguiente_sesion)
    ops = ops.dropna(subset=["fecha_trade"])

    for r in ops.itertuples():
        precios.loc[r.fecha_trade, r.ticker] = r.precio

    precios = precios.ffill()

    return precios, dividendos, ops


def historico_valor_cartera(archivo, fecha_fin=None, capital_inicial=10_000_000,
                            hoja="Operativa", incluir_costes=True):

    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin)

    posiciones = pd.DataFrame(0.0, index=precios.index, columns=precios.columns)
    flujos = pd.Series(0.0, index=precios.index)

    for r in ops.itertuples():
        signo = 1 if r.accion == "COMPRA" else -1
        cantidad = abs(float(r.cantidad))

        posiciones.loc[r.fecha_trade:, r.ticker] += signo * cantidad
        flujos.loc[r.fecha_trade] -= signo * cantidad * float(r.precio_ejecutado)

    dividendos_diarios = (posiciones.shift(1).fillna(0.0) * dividendos).sum(axis=1)
    cash = capital_inicial + flujos.cumsum() + dividendos_diarios.cumsum()
    valor_acciones = (posiciones * precios).sum(axis=1)
    valor_cartera = cash + valor_acciones

    return pd.DataFrame({
        "Cash": cash,
        "Valor acciones": valor_acciones,
        "Dividendos diarios": dividendos_diarios,
        "Valor cartera": valor_cartera,
        "Rentabilidad diaria": valor_cartera.pct_change().fillna(0.0),
    })


def tabla_semanal_atribucion(archivo, universo_tickers, fecha_fin=None,
                             capital_inicial=10_000_000, hoja="Operativa",
                             incluir_costes=True):

    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin, universo_tickers)

    fecha_valor = precios.index[precios.index <= fecha_fin].max()
    fechas_op = sorted(f for f in ops["fecha_trade"].unique() if f <= fecha_valor)

    pos = {}
    cash = float(capital_inicial)
    filas = []

    def valor_posicion(fecha):
        return sum(
            q * precios.loc[fecha, t]
            for t, q in pos.items()
            if t in precios.columns and pd.notna(precios.loc[fecha, t])
        )

    def retorno_total(t, f0, f1, divs):
        p0, p1 = precios.loc[f0, t], precios.loc[f1, t]
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            return np.nan
        return (p1 + divs[t].sum()) / p0 - 1

    for i, f0 in enumerate(fechas_op):
        f1 = fechas_op[i + 1] if i + 1 < len(fechas_op) else fecha_valor
        nav0 = cash + valor_posicion(f0)

        if nav0 <= 0 or pd.isna(nav0):
            continue

        cash_sin_costes = cash

        for r in ops[ops["fecha_trade"].eq(f0)].itertuples():
            signo = 1 if r.accion == "COMPRA" else -1
            cantidad = abs(float(r.cantidad))

            pos[r.ticker] = pos.get(r.ticker, 0.0) + signo * cantidad
            cash_sin_costes -= signo * cantidad * float(r.precio)
            cash -= signo * cantidad * float(r.precio_ejecutado)

        pos = {t: q for t, q in pos.items() if abs(q) > 1e-12}

        coste_eur = cash_sin_costes - cash
        coste = -coste_eur / nav0

        divs = dividendos.loc[(dividendos.index > f0) & (dividendos.index <= f1)]

        ret_universo = pd.Series({
            t: retorno_total(t, f0, f1, divs)
            for t in universo_tickers
            if t in precios.columns
        }).dropna()

        r_bmk = float(ret_universo.mean()) if len(ret_universo) else 0.0
        w_bmk = 1 / len(ret_universo) if len(ret_universo) else 0.0

        seleccion = 0.0
        pesos = 0.0

        for t, q in pos.items():
            if t not in precios.columns:
                continue

            r = retorno_total(t, f0, f1, divs)
            p0 = precios.loc[f0, t]

            if pd.isna(r) or pd.isna(p0):
                continue

            w = q * p0 / nav0
            b = w_bmk if t in ret_universo.index else 0.0
            exceso = r - r_bmk

            seleccion += b * exceso
            pesos += (w - b) * exceso

        peso_cash = cash_sin_costes / nav0
        pesos += peso_cash * (0.0 - r_bmk)

        divs_eur = sum(q * divs[t].sum() for t, q in pos.items() if t in divs.columns)

        nav1 = cash + divs_eur + valor_posicion(f1)
        r_cartera = nav1 / nav0 - 1

        filas.append({
            "Periodo": f"{pd.Timestamp(f0).date()} → {pd.Timestamp(f1).date()}",
            "NAV inicial": nav0,
            "NAV final": nav1,
            "Rent. cartera neta": r_cartera,
            "Rent. BMK EW": r_bmk,
            "Alpha": r_cartera - r_bmk,
            "Ef. selección": seleccion,
            "Ef. pesos": pesos,
            "Costes": coste,
            "Costes (€)": -coste_eur,
            "Dividendos (€)": divs_eur,
            "Check": r_cartera - (r_bmk + seleccion + pesos + coste),
        })

        cash += divs_eur

    return pd.DataFrame(filas).set_index("Periodo")


def encadenar_atribucion(semanal, capital_inicial=10_000_000):
    nav = float(capital_inicial)
    bmk = float(capital_inicial)

    comp = {
        "Ef. selección (€)": 0.0,
        "Ef. pesos (€)": 0.0,
        "Costes (€)": 0.0,
    }

    filas = []

    for periodo, row in semanal.iterrows():
        r_cartera = float(row["Rent. cartera neta"])
        r_bmk = float(row["Rent. BMK EW"])
        bmk_ini = bmk

        efectos = {
            "Ef. selección (€)": float(row["Ef. selección"]),
            "Ef. pesos (€)": float(row["Ef. pesos"]),
            "Costes (€)": float(row["Costes"]),
        }

        for nombre, efecto in efectos.items():
            comp[nombre] = comp[nombre] * (1 + r_cartera) + bmk_ini * efecto

        nav *= 1 + r_cartera
        bmk *= 1 + r_bmk

        filas.append({
            "Periodo": periodo,
            "NAV cartera": nav,
            "NAV BMK EW": bmk,
            "Alpha acumulado (€)": nav - bmk,
            **comp,
            "Check alpha (€)": sum(comp.values()) - (nav - bmk),
        })

    final = pd.Series({
        "NAV cartera": nav,
        "NAV BMK EW": bmk,
        "Rentabilidad cartera": nav / capital_inicial - 1,
        "Rentabilidad BMK EW": bmk / capital_inicial - 1,
        "Resultado cartera (€)": nav - capital_inicial,
        "Efecto mercado (€)": bmk - capital_inicial,
        "Alpha (€)": nav - bmk,
        **comp,
    })

    final["Check alpha (€)"] = (
        final["Ef. selección (€)"]
        + final["Ef. pesos (€)"]
        + final["Costes (€)"]
        - final["Alpha (€)"]
    )

    final["Check NAV (€)"] = (
        final["Efecto mercado (€)"]
        + final["Ef. selección (€)"]
        + final["Ef. pesos (€)"]
        + final["Costes (€)"]
        - final["Resultado cartera (€)"]
    )

    return pd.DataFrame(filas).set_index("Periodo"), final


def formatear_atribucion(semanal, detalle_acumulado, final):
    def pct(x):
        return f"{float(x):.2%}" if pd.notna(x) else ""

    def keur(x):
        return f"{float(x) / 1000:,.1f} k€" if pd.notna(x) else ""

    semanal_fmt = semanal.copy().astype(object)
    detalle_fmt = detalle_acumulado.copy().astype(object)
    final_fmt = final.to_frame("Valor").astype(object)

    pct_cols = [
        "Rent. cartera neta", "Rent. BMK EW", "Alpha",
        "Ef. selección", "Ef. pesos", "Costes", "Check"
    ]

    eur_cols = [
        "NAV inicial", "NAV final", "Costes (€)", "Dividendos (€)"
    ]

    for c in pct_cols:
        if c in semanal_fmt.columns:
            semanal_fmt[c] = semanal[c].map(pct)

    for c in eur_cols:
        if c in semanal_fmt.columns:
            semanal_fmt[c] = semanal[c].map(keur)

    for c in detalle_fmt.columns:
        if "(€)" in c or "NAV" in c:
            detalle_fmt[c] = detalle_acumulado[c].map(keur)

    for idx in final_fmt.index:
        valor = final.loc[idx]

        if "(€)" in idx or "NAV" in idx:
            final_fmt.loc[idx, "Valor"] = keur(valor)
        elif "Rentabilidad" in idx:
            final_fmt.loc[idx, "Valor"] = pct(valor)
        else:
            final_fmt.loc[idx, "Valor"] = valor

    return semanal_fmt, detalle_fmt, final_fmt


def series_diarias_cartera_bmks(archivo, universo_tickers, fecha_fin=None,
                                capital_inicial=10_000_000, hoja="Operativa",
                                incluir_costes=True, benchmark="^STOXX50E"):

    semanal = tabla_semanal_atribucion(
        archivo=archivo,
        universo_tickers=universo_tickers,
        fecha_fin=fecha_fin,
        capital_inicial=capital_inicial,
        hoja=hoja,
        incluir_costes=incluir_costes,
    )

    _, final = encadenar_atribucion(semanal, capital_inicial)

    hist = historico_valor_cartera(
        archivo=archivo,
        fecha_fin=fecha_fin,
        capital_inicial=capital_inicial,
        hoja=hoja,
        incluir_costes=incluir_costes,
    )

    fecha_fin_real = pd.to_datetime(semanal.index[-1].split(" → ")[1])
    cartera = hist.loc[:fecha_fin_real, "Valor cartera"].copy()
    cartera.name = "Estrategia real"

    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, _ = _datos_mercado(ops, fecha_fin, universo_tickers)
    precios = precios.loc[cartera.index.min():fecha_fin_real]
    dividendos = dividendos.reindex(precios.index).fillna(0.0)

    bmk_ew = _serie_bmk_ew_diaria(
        semanal=semanal,
        precios=precios,
        dividendos=dividendos,
        universo_tickers=universo_tickers,
        capital_inicial=capital_inicial,
    )
    bmk_ew = bmk_ew.reindex(cartera.index).ffill()
    bmk_ew.name = "STOXX 50 EW"

    datos_bmk = yf.download(
        benchmark,
        start=cartera.index.min().strftime("%Y-%m-%d"),
        end=(fecha_fin_real + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    bmk = datos_bmk["Close"].squeeze().reindex(cartera.index).ffill()
    bmk = capital_inicial * bmk / bmk.iloc[0]
    bmk.name = "STOXX 50"

    series = pd.concat([cartera, bmk, bmk_ew], axis=1).dropna(how="all")

    if abs(series["Estrategia real"].iloc[-1] - final["NAV cartera"]) > 1:
        raise ValueError("La cartera diaria no cuadra con el NAV final.")

    if abs(series["STOXX 50 EW"].iloc[-1] - final["NAV BMK EW"]) > 1:
        raise ValueError("El benchmark EW diario no cuadra con el NAV BMK EW final.")

    retornos = series.pct_change().dropna()

    tabla_metricas = pd.DataFrame(index=series.columns)
    tabla_metricas["Rentabilidad"] = series.iloc[-1] / series.iloc[0] - 1
    tabla_metricas["Volatilidad"] = retornos.std() * np.sqrt(252)
    tabla_metricas["Max DD"] = (series / series.cummax() - 1).min()
    tabla_metricas["Sharpe"] = (retornos.mean() / retornos.std()) * np.sqrt(252)

    tabla_metricas = tabla_metricas.reset_index().rename(columns={"index": "Estrategia"})

    tabla_metricas_fmt = (
        tabla_metricas.style
        .format({
            "Rentabilidad": "{:.2%}",
            "Volatilidad": "{:.2%}",
            "Max DD": "{:.2%}",
            "Sharpe": "{:.2f}",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#28669A"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #8AA6C1"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "white"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #E0E0E0"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-weight", "bold"),
                    ("font-size", "14px"),
                    ("color", "#1E2A4A"),
                ],
            },
        ])
        .hide(axis="index")
    )

    return series, semanal, final, tabla_metricas, tabla_metricas_fmt


def _serie_bmk_ew_diaria(semanal, precios, dividendos, universo_tickers,
                         capital_inicial=10_000_000):

    serie = {}
    valor_base = float(capital_inicial)

    for periodo, row in semanal.iterrows():
        f0_txt, f1_txt = periodo.split(" → ")
        f0, f1 = pd.Timestamp(f0_txt), pd.Timestamp(f1_txt)

        fechas = precios.index[(precios.index >= f0) & (precios.index <= f1)]
        if len(fechas) == 0:
            continue

        tickers = [t for t in universo_tickers if t in precios.columns]
        p0 = precios.loc[f0, tickers].dropna()
        p1 = precios.loc[f1, p0.index].dropna()
        validos = p0.index.intersection(p1.index)
        validos = [t for t in validos if p0[t] != 0]

        if not validos:
            continue

        divs = dividendos.loc[fechas, validos].copy()
        divs.loc[divs.index <= f0, :] = 0.0
        divs_acum = divs.cumsum()

        rel = (precios.loc[fechas, validos] + divs_acum).div(p0[validos]) - 1
        valores = valor_base * (1 + rel.mean(axis=1))

        serie.update(valores.to_dict())
        valor_base *= 1 + float(row["Rent. BMK EW"])

    return pd.Series(serie).sort_index()


def grafico_diario_cartera_bmks(series, titulo="Evolución diaria de la cartera vs benchmarks",
                                guardar=None):

    col_cartera = "Estrategia real"

    colores = {
        "Estrategia real": "#4F82FF",
        "STOXX 50": "#FFB84D",
        "STOXX 50 EW": "#2FE6D0",
    }

    fondo = "#070A2D"
    grid = "#2A2F5F"
    texto = "white"

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=180)
    fig.patch.set_facecolor(fondo)
    ax.set_facecolor(fondo)

    for col in series.columns:
        ax.plot(
            series.index,
            series[col],
            label=col,
            color=colores.get(col, "white"),
            linewidth=3.0 if col == col_cartera else 2.3,
            alpha=0.98,
        )

    ax.axhline(
        series.iloc[0, 0],
        color="white",
        linewidth=1.2,
        linestyle="--",
        alpha=0.35,
    )

    ax.fill_between(
        series.index,
        series[col_cartera],
        series.iloc[0, 0],
        color=colores[col_cartera],
        alpha=0.10,
    )

    for col in series.columns:
        y = series[col].iloc[-1]
        ax.scatter(series.index[-1], y, color=colores.get(col, "white"), s=45, zorder=5)
        ax.text(
            series.index[-1],
            y,
            f"  {y / 1_000_000:.2f} M€",
            color=colores.get(col, "white"),
            fontsize=12,
            fontweight="bold",
            va="center",
        )

    ax.set_title(titulo, color=texto, fontsize=21, fontweight="bold", pad=20)
    ax.set_ylabel("Valor de la cartera", color=texto, fontsize=14, fontweight="bold")

    ax.tick_params(axis="x", colors=texto, labelsize=12, rotation=25)
    ax.tick_params(axis="y", colors=texto, labelsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.2f} M€"))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#6D739C")
    ax.spines["bottom"].set_color("#6D739C")
    ax.grid(True, color=grid, linewidth=0.8, alpha=0.45)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#101545",
        edgecolor="#3C4275",
        fontsize=12,
    )

    for text in legend.get_texts():
        text.set_color("white")

    rent_cartera = series[col_cartera].iloc[-1] / series[col_cartera].iloc[0] - 1
    alpha_ew = series[col_cartera].iloc[-1] - series["STOXX 50 EW"].iloc[-1]

    # caja = (
    #     f"NAV final: {series[col_cartera].iloc[-1] / 1_000_000:.3f} M€\n"
    #     f"Rentabilidad: {rent_cartera:+.2%}\n"
    #     f"Alpha vs EW: {alpha_ew / 1000:+.0f} k€"
    # )

    # ax.text(
    #     0.02,
    #     0.08,
    #     caja,
    #     transform=ax.transAxes,
    #     fontsize=13,
    #     color="white",
    #     fontweight="bold",
    #     bbox=dict(
    #         boxstyle="round,pad=0.55,rounding_size=0.18",
    #         facecolor="#2F6EA7",
    #         edgecolor="none",
    #         alpha=0.95,
    #     ),
    # )

    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()

    return fig, ax

