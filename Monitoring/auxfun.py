import pandas as pd
import yfinance as yf
import unicodedata
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _capar_pesos(w, cap=0.10):
    w = w.dropna()
    libres = list(w.index)
    out = pd.Series(0.0, index=w.index)
    restante = 1.0

    while libres:
        aux = w[libres] / w[libres].sum() * restante
        topados = aux[aux > cap].index

        if len(topados) == 0:
            out[libres] = aux
            break

        out[topados] = cap
        restante -= cap * len(topados)
        libres = [t for t in libres if t not in topados]

    return out / out.sum()


def calcular_pesos_bmk_actuales(tickers, cap=0.10):
    caps = {}

    for t in tickers:
        tk = yf.Ticker(t)
        try:
            mc = tk.fast_info.get("market_cap", np.nan)
        except Exception:
            mc = np.nan

        if pd.isna(mc):
            try:
                mc = tk.info.get("marketCap", np.nan)
            except Exception:
                mc = np.nan

        if pd.notna(mc) and mc > 0:
            caps[t] = float(mc)

    if not caps:
        raise ValueError("No se han podido calcular capitalizaciones para el benchmark.")

    w = pd.Series(caps, dtype=float)
    return _capar_pesos(w / w.sum(), cap=cap)


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
    
    precios = precios.loc[:fecha_fin].copy()
    dividendos = dividendos.reindex(precios.index).fillna(0.0)

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


def _serie_benchmark(benchmark, fechas, start, end):
    datos = yf.download(
        benchmark,
        start=(pd.to_datetime(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        end=(pd.to_datetime(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if datos.empty:
        raise ValueError(f"No se han podido descargar datos del benchmark {benchmark}.")

    serie = datos["Close"].squeeze()
    serie.index = pd.to_datetime(serie.index).tz_localize(None)
    fechas = pd.DatetimeIndex(pd.to_datetime(fechas)).tz_localize(None)

    serie = serie.reindex(serie.index.union(fechas)).sort_index().ffill().reindex(fechas)

    if serie.isna().any():
        raise ValueError(f"Faltan precios del benchmark {benchmark} para alguna fecha.")

    return serie


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


def pnl_por_activo(archivo, fecha_fin=None, capital_inicial=10_000_000,
                   hoja="Operativa", incluir_costes=True, nombres_cortos=None):

    nombres_cortos = nombres_cortos or {
        "SAN.MC": "B. Santander", "BBVA.MC": "BBVA", "BNP.PA": "BNP",
        "INGA.AS": "ING", "ADS.DE": "Adidas", "ASML.AS": "ASML",
        "IFX.DE": "Infineon", "ENR.DE": "Siemens Energy", "UCG.MI": "UniCredit",
        "SAF.PA": "Safran", "SGO.PA": "Saint-Gobain", "RACE.MI": "Ferrari",
        "RMS.PA": "Hermès", "ARGX.BR": "Argenx", "ENI.MI": "ENI",
        "ITX.MC": "Inditex", "AIR.PA": "Airbus", "ADYEN.AS": "Adyen",
        "EL.PA": "EssilorLuxottica", "WKL.AS": "Wolters Kluwer",
        "PRX.AS": "Prosus", "SAP.DE": "SAP", "BAYN.DE": "Bayer",
        "TTE.PA": "TotalEnergies", "RHM.DE": "Rheinmetall",
        "DBK.DE": "Deutsche Bank",
    }

    def nombre(t):
        if t in nombres_cortos:
            return nombres_cortos[t]
        try:
            n = yf.Ticker(t).info.get("shortName", t)
        except Exception:
            n = t
        return str(n).replace(" S.A.", "").replace(" SE", "").replace(" AG", "").replace(" N.V.", "")[:22]

    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin)

    fecha_valor = precios.index[precios.index <= fecha_fin].max()
    ops = ops[ops["fecha_trade"] <= fecha_valor].copy()
    tickers = sorted(ops["ticker"].unique())

    precios = precios.loc[:fecha_valor, tickers]
    dividendos = dividendos.reindex(precios.index).fillna(0.0)[tickers]

    posiciones = pd.DataFrame(0.0, index=precios.index, columns=tickers)

    for r in ops.itertuples():
        signo = 1 if r.accion == "COMPRA" else -1
        posiciones.loc[r.fecha_trade:, r.ticker] += signo * abs(float(r.cantidad))

    ops["flujo"] = np.where(ops["accion"].eq("VENTA"), 1, -1) * ops["cantidad"].abs() * ops["precio_ejecutado"]

    pnl = ops.groupby("ticker")["flujo"].sum().reindex(tickers, fill_value=0.0)
    pnl += (posiciones.shift(1).fillna(0.0) * dividendos).sum()
    pnl += posiciones.loc[fecha_valor] * precios.loc[fecha_valor]

    total = float(pnl.sum())

    cash_final = capital_inicial + ops["flujo"].sum() + (posiciones.shift(1).fillna(0.0) * dividendos).sum().sum()
    nav_real = cash_final + (posiciones.loc[fecha_valor] * precios.loc[fecha_valor]).sum()

    if abs((capital_inicial + total) - nav_real) > 1:
        raise ValueError("El P&L por activo no cuadra con el NAV final.")

    tabla = pnl.rename("P&L (€)").to_frame()
    tabla["Activo"] = tabla.index.map(nombre)
    tabla = tabla.set_index("Activo").sort_values("P&L (€)")
    tabla.loc["TOTAL"] = total

    fondo = "#070A2D"
    datos = tabla.drop("TOTAL")
    datos = datos[
        ~datos.index.to_series().str.contains("overn|swap|rate", case=False, regex=True)
    ]
    pnl_medio = float(datos["P&L (€)"].mean())
    print(f"P&L medio por activo: {pnl_medio:,.2f} €")
    colores = np.where(datos["P&L (€)"] >= 0, "#2FE6D0", "#FF3B30")

    fig, ax = plt.subplots(figsize=(12.5, max(5, 0.32 * len(datos))), dpi=180)
    fig.patch.set_facecolor(fondo)
    ax.set_facecolor(fondo)

    ax.barh(datos.index, datos["P&L (€)"], color=colores, alpha=0.92)
    ax.axvline(0, color="white", linewidth=1, alpha=0.65)
    ax.axvline(total, color="#FFD166", linewidth=2.2, linestyle="--",
               label=f"P&L total: {total:,.0f} €")

    ax.set_title("P&L por activo", color="white", fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("P&L neto (€)", color="white", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f} €"))

    ax.tick_params(colors="white", labelsize=10)
    ax.grid(True, axis="x", color="white", alpha=0.12, linewidth=0.8)

    for s in ax.spines.values():
        s.set_color("#6D739C")

    leg = ax.legend(loc="lower right", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("#101545")
    leg.get_frame().set_edgecolor("#4D5AA0")
    leg.get_frame().set_alpha(0.85)
    for txt in leg.get_texts():
        txt.set_color("white")

    plt.tight_layout()
    plt.show()

    return tabla, fig, ax


def tabla_semanal_atribucion(archivo, universo_tickers, fecha_fin=None,
                             capital_inicial=10_000_000, hoja="Operativa",
                             incluir_costes=True, pesos_bmk=None,
                             benchmark="^STOXX50E"):

    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin, universo_tickers)

    if pesos_bmk is None:
        pesos_bmk = calcular_pesos_bmk_actuales(universo_tickers)

    pesos_bmk = pd.Series(pesos_bmk, dtype=float)
    pesos_bmk = pesos_bmk[pesos_bmk > 0] / pesos_bmk[pesos_bmk > 0].sum()

    fecha_valor = precios.index[precios.index <= fecha_fin].max()
    fechas_op = sorted(f for f in ops["fecha_trade"].unique() if f <= fecha_valor)

    if len(fechas_op) == 0:
        raise ValueError("No hay fechas de operación válidas para la atribución.")

    fechas_bmk = sorted(set(fechas_op + [fecha_valor]))
    serie_bmk = _serie_benchmark(
        benchmark=benchmark,
        fechas=fechas_bmk,
        start=min(fechas_bmk),
        end=max(fechas_bmk),
    )

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

        if nav0 <= 0 or pd.isna(nav0) or f0 == f1:
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

        r_bmk = float(serie_bmk.loc[f1] / serie_bmk.loc[f0] - 1)

        ret_universo = pd.Series({
            t: retorno_total(t, f0, f1, divs)
            for t in universo_tickers
            if t in precios.columns
        }).dropna()

        pesos_periodo = pesos_bmk.reindex(ret_universo.index).dropna()
        if len(pesos_periodo):
            pesos_periodo = pesos_periodo / pesos_periodo.sum()
        else:
            pesos_periodo = pd.Series(dtype=float)

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
            b = float(pesos_periodo.get(t, 0.0))
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
            "Rent. BMK": r_bmk,
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
        r_bmk = float(row["Rent. BMK"])
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
            "NAV BMK": bmk,
            "Alpha acumulado (€)": nav - bmk,
            **comp,
            "Check alpha (€)": sum(comp.values()) - (nav - bmk),
        })

    final = pd.Series({
        "NAV cartera": nav,
        "NAV BMK": bmk,
        "Rentabilidad cartera": nav / capital_inicial - 1,
        "Rentabilidad BMK": bmk / capital_inicial - 1,
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


def formatear_atribucion(semanal, final):
    def pct(x):
        return "" if pd.isna(x) else f"{float(x):.2%}"

    def keur(x):
        return "" if pd.isna(x) else f"{float(x) / 1000:,.1f} k€"

    def num(x):
        return "" if pd.isna(x) else f"{float(x):.2f}"

    def vol_anual(r):
        return r.std(ddof=1) * np.sqrt(52)

    def sharpe(r):
        v = r.std(ddof=1)
        return np.nan if v == 0 or pd.isna(v) else r.mean() / v * np.sqrt(52)

    cols = [
        "NAV inicial", "NAV final", "Rent. cartera neta",
        "Alpha", "Ef. selección", "Ef. pesos", "Costes"
    ]

    semanal_fmt = semanal[[c for c in cols if c in semanal.columns]].copy().astype(object)

    for c in ["Rent. cartera neta", "Alpha", "Ef. selección", "Ef. pesos", "Costes"]:
        if c in semanal_fmt.columns:
            semanal_fmt[c] = semanal[c].map(pct)

    for c in ["NAV inicial", "NAV final"]:
        if c in semanal_fmt.columns:
            semanal_fmt[c] = semanal[c].map(keur)

    r_cartera = semanal["Rent. cartera neta"]
    r_bmk = semanal["Rent. BMK"]
    capital_ini = final["NAV cartera"] - final["Resultado cartera (€)"]

    final_fmt = pd.DataFrame({
        "Cartera": {
            "NAV final": keur(final["NAV cartera"]),
            "Resultado": f"{keur(final['Resultado cartera (€)'])} ({pct(final['Rentabilidad cartera'])})",
            "Rentabilidad": pct(final["Rentabilidad cartera"]),
            "Volatilidad anualizada": pct(vol_anual(r_cartera)),
            "Sharpe": num(sharpe(r_cartera)),
            "Alpha vs BMK": f"{keur(final['Alpha (€)'])} ({pct(final['Alpha (€)'] / capital_ini)})",
            "Ef. selección": f"{keur(final['Ef. selección (€)'])} ({pct(final['Ef. selección (€)'] / capital_ini)})",
            "Ef. pesos": f"{keur(final['Ef. pesos (€)'])} ({pct(final['Ef. pesos (€)'] / capital_ini)})",
            "Costes": f"{keur(final['Costes (€)'])} ({pct(final['Costes (€)'] / capital_ini)})",
        },
        "BMK": {
            "NAV final": keur(final["NAV BMK"]),
            "Resultado": f"{keur(final['Efecto mercado (€)'])} ({pct(final['Rentabilidad BMK'])})",
            "Rentabilidad": pct(final["Rentabilidad BMK"]),
            "Volatilidad anualizada": pct(vol_anual(r_bmk)),
            "Sharpe": num(sharpe(r_bmk)),
            "Alpha vs BMK": "",
            "Ef. selección": "",
            "Ef. pesos": "",
            "Costes": "",
        }
    })

    return semanal_fmt, final_fmt


def series_diarias_cartera_bmks(archivo, universo_tickers, fecha_fin=None,
                                capital_inicial=10_000_000, hoja="Operativa",
                                incluir_costes=True, benchmark="^STOXX50E", rf_anual=0.02):

    semanal = tabla_semanal_atribucion(
        archivo=archivo,
        universo_tickers=universo_tickers,
        fecha_fin=fecha_fin,
        capital_inicial=capital_inicial,
        hoja=hoja,
        incluir_costes=incluir_costes,
        benchmark=benchmark,
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

    series = pd.concat([cartera, bmk], axis=1).dropna(how="all")

    if abs(series["Estrategia real"].iloc[-1] - final["NAV cartera"]) > 1:
        raise ValueError("La cartera diaria no cuadra con el NAV final.")

    retornos = series.pct_change().dropna()
    rf_diario = (1 + rf_anual) ** (1 / 252) - 1
    exceso = retornos - rf_diario

    tabla_metricas = pd.DataFrame(index=series.columns)
    tabla_metricas["Rentabilidad"] = [
        series["Estrategia real"].iloc[-1] / capital_inicial - 1,
        series["STOXX 50"].iloc[-1] / capital_inicial - 1,
    ]
    tabla_metricas["Volatilidad"] = retornos.std() * np.sqrt(252)
    tabla_metricas["Max DD"] = (series / series.cummax() - 1).min()
    tabla_metricas["Sharpe"] = (exceso.mean() / retornos.std()) * np.sqrt(252)

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


def grafico_evolucion_drawdown(series, titulo="Evolución de la cartera y drawdown"):
    cols = [c for c in ["Estrategia real", "STOXX 50"] if c in series.columns]
    dd = series[cols].div(series[cols].cummax()).sub(1)

    fondo = "#070A2D"
    colores = {"Estrategia real": "#4F82FF", "STOXX 50": "#FFB84D"}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08}
    )
    fig.patch.set_facecolor(fondo)

    for ax in (ax1, ax2):
        ax.set_facecolor(fondo)
        ax.grid(True, color="white", alpha=0.12, linewidth=0.8)
        ax.tick_params(colors="white", labelsize=11)
        for s in ax.spines.values():
            s.set_color("#6D739C")

    for col in cols:
        ax1.plot(series.index, series[col], color=colores[col], linewidth=2.7, label=col)
        ax2.plot(dd.index, dd[col], color=colores[col], linewidth=2.0,
                 label=f"{col} ({dd[col].min():.2%})")

    ax2.fill_between(dd.index, dd["Estrategia real"], 0, color="#FF3B30", alpha=0.28)
    ax1.axhline(series.iloc[0, 0], color="white", linestyle="--", linewidth=1, alpha=0.35)
    ax2.axhline(0, color="white", linewidth=1, alpha=0.65)

    ax1.set_title(titulo, color="white", fontsize=22, fontweight="bold", pad=16)
    ax1.set_ylabel("Valor cartera", color="white", fontsize=13)
    ax2.set_ylabel("Drawdown", color="white", fontsize=13)
    ax2.set_xlabel("Fecha", color="white", fontsize=13)

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.2f} M€"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.set_ylim(dd.min().min() * 1.15, 0.003)

    for ax, loc, fs in [(ax1, "upper left", 11), (ax2, "lower left", 10)]:
        leg = ax.legend(loc=loc, frameon=True, fontsize=fs)
        leg.get_frame().set_facecolor("#101545")
        leg.get_frame().set_edgecolor("#4D5AA0")
        leg.get_frame().set_alpha(0.82)
        for t in leg.get_texts():
            t.set_color("white")

    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)


def grafico_evolucion_drawdown_plotly(series, titulo="Evolución de la cartera y drawdown"):
    cols = [c for c in ["Estrategia real", "STOXX 50"] if c in series.columns]
    dd = series[cols].div(series[cols].cummax()).sub(1)

    fondo = "#070A2D"
    colores = {"Estrategia real": "#4F82FF", "STOXX 50": "#FFB84D"}

    # 1. Crear la estructura base (equivalente a plt.subplots y gridspec_kw)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32] # Aprox ratio 2.1 a 1
    )

    # 2. Trazos del NAV (Gráfico superior)
    for col in cols:
        # Dividimos entre 1M para facilitar el formateo del eje Y en formato estático
        nav_millions = series[col] / 1_000_000 
        fig.add_trace(
            go.Scatter(
                x=series.index, y=nav_millions, 
                mode="lines", name=col,
                line=dict(color=colores[col], width=2.7)
            ),
            row=1, col=1
        )
    
    # Línea base del NAV inicial. Al usar iloc[0, 0], respetamos dinámicamente el valor de 
    # arranque real de tu serie (sea el del 12 de marzo u otro) sin fijarlo estáticamente en 10M€.
    nav_inicial = series.iloc[0, 0] / 1_000_000
    fig.add_hline(y=nav_inicial, line_dash="dash", line_color="white", 
                  line_width=1, opacity=0.35, row=1, col=1)

    # 3. Trazos del Drawdown (Gráfico inferior)
    for col in cols:
        # Lógica para replicar el fill_between solo en la estrategia real
        fill = 'tozeroy' if col == "Estrategia real" else 'none'
        fillcolor = 'rgba(255, 59, 48, 0.28)' if col == "Estrategia real" else None
        
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd[col], 
                mode="lines", name=f"{col} ({dd[col].min():.2%})",
                line=dict(color=colores[col], width=2.0),
                fill=fill, fillcolor=fillcolor
            ),
            row=2, col=1
        )

    # Línea base de cero para el DD
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.65, row=2, col=1)

    # 4. Configuración global del Layout (Fondos, título y leyenda)
    fig.update_layout(
        title=dict(text=titulo, font=dict(color="white", size=22), pad=dict(b=16)),
        plot_bgcolor=fondo,
        paper_bgcolor=fondo,
        margin=dict(l=60, r=40, t=80, b=50), # Equivalente a plt.tight_layout()
        legend=dict(
            bgcolor="rgba(16, 21, 69, 0.82)", # "#101545" con transparencia
            bordercolor="#4D5AA0", borderwidth=1,
            font=dict(color="white", size=11),
            # En Plotly, agrupar las dos leyendas suele quedar mejor arriba a la izquierda
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified" # Esto añade una línea vertical interactiva muy analítica
    )

    # 5. Configuración de Ejes (Equivalente a ax.tick_params, ax.grid, spines y set_major_formatter)
    grid_style = dict(showgrid=True, gridcolor="rgba(255,255,255,0.12)", gridwidth=0.8)
    axis_style = dict(tickfont=dict(color="white", size=11), showline=True, linecolor="#6D739C", linewidth=1)

    fig.update_xaxes(**grid_style, **axis_style, row=1, col=1)
    fig.update_xaxes(title_text="Fecha", title_font=dict(color="white", size=13), 
                     **grid_style, **axis_style, row=2, col=1)

    fig.update_yaxes(title_text="Valor cartera", title_font=dict(color="white", size=13),
                     tickformat=".2f", ticksuffix=" M€", 
                     **grid_style, **axis_style, row=1, col=1)
                     
    fig.update_yaxes(title_text="Drawdown", title_font=dict(color="white", size=13),
                     tickformat=".0%", range=[dd.min().min() * 1.15, 0.003],
                     **grid_style, **axis_style, row=2, col=1)

    fig.show()
    return fig