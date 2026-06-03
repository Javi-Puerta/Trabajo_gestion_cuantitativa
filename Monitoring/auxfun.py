import pandas as pd
import yfinance as yf
import unicodedata
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Este tema es para la presentación
TEMA_PRESENTACION = {
    "fondo": "#070A2D", "panel": "#06133A", "header": "#28669A",
    "lineas": "#27456F", "blanco": "#FFFFFF", "positivo": "#2FE6D0",
    "negativo": "#FF6B6B", "naranja": "#FFB84D", "azul": "#4F82FF",
}

# Este tema es para la memoria del proyecto
TEMA_MEMORIA = {
    "fondo": "#FFFFFF",
    "panel": "#F6F8FB",
    "header": "#1F4E79",

    "texto": "#111827",
    "texto_suave": "#4B5563",
    "blanco": "#FFFFFF",

    "lineas": "#C8D2E0",
    "grid": "#E5E7EB",

    "azul": "#2563EB",
    "naranja": "#D97706",

    "positivo": "#059669",
    "negativo": "#DC2626",
    "destacado": "#0F766E",
}


def get_eurostoxx50_tickers():
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 ..."}, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text), flavor="bs4")
    return next(t for t in tables if "Ticker" in t.columns)["Ticker"].tolist()


def _limpiar_col(c):
    c = unicodedata.normalize("NFKD", str(c))
    return "".join(x for x in c if not unicodedata.combining(x)).lower().strip().replace(" ", "_")


def _leer_operaciones(archivo, fecha_fin=None, hoja="Operativa", incluir_costes=True):
    ops = pd.read_excel(archivo, sheet_name=hoja) if str(archivo).endswith((".xlsx", ".xls")) else pd.read_csv(archivo)
    ops.columns = [_limpiar_col(c) for c in ops.columns]
    ops = ops.rename(columns={"id": "ticker"})
    ops["fecha"] = pd.to_datetime(ops["fecha"], dayfirst=True)
    ops["accion"] = ops["accion"].astype(str).str.upper().str.strip()
    ops = ops[ops["accion"].isin(["COMPRA", "VENTA"]) & ops["ticker"].notna()].copy()
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
    libres, out, restante = list(w.index), pd.Series(0.0, index=w.index), 1.0
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


def _datos_mercado(ops, fecha_fin, universo_tickers=(), lookback_dias=0):
    tickers = sorted(set(ops["ticker"]) | set(universo_tickers))
    start, end = ops["fecha"].min() - pd.Timedelta(days=lookback_dias), fecha_fin + pd.Timedelta(days=2)
    if start >= end:
        raise ValueError(f"Fechas inválidas: start={start.date()} >= end={end.date()}")

    datos = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                        actions=True, auto_adjust=False, progress=False)
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
    return precios.ffill(), dividendos, ops


def _serie_benchmark(benchmark, fechas, start, end):
    datos = yf.download(benchmark, start=(pd.to_datetime(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                        end=(pd.to_datetime(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                        auto_adjust=True, progress=False)
    if datos.empty:
        raise ValueError(f"No se han podido descargar datos del benchmark {benchmark}.")

    serie = datos["Close"].squeeze()
    serie.index = pd.to_datetime(serie.index).tz_localize(None)
    fechas = pd.DatetimeIndex(pd.to_datetime(fechas)).tz_localize(None)
    serie = serie.reindex(serie.index.union(fechas)).sort_index().ffill().reindex(fechas)
    if serie.isna().any():
        raise ValueError(f"Faltan precios del benchmark {benchmark} para alguna fecha.")
    return serie


def _reconstruir_cartera_diaria(archivo, fecha_fin=None, capital_inicial=10_000_000, hoja="Operativa",
                                incluir_costes=True, universo_tickers=(), lookback_dias=0):
    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin, universo_tickers, lookback_dias)

    posiciones = pd.DataFrame(0.0, index=precios.index, columns=precios.columns)
    flujos = pd.Series(0.0, index=precios.index)
    for r in ops.itertuples():
        signo, cantidad = (1 if r.accion == "COMPRA" else -1), abs(float(r.cantidad))
        posiciones.loc[r.fecha_trade:, r.ticker] += signo * cantidad
        flujos.loc[r.fecha_trade] -= signo * cantidad * float(r.precio_ejecutado)

    dividendos_diarios = (posiciones.shift(1).fillna(0.0) * dividendos).sum(axis=1)
    cash = capital_inicial + flujos.cumsum() + dividendos_diarios.cumsum()
    valor_posiciones = posiciones * precios
    valor_cartera = cash + valor_posiciones.sum(axis=1)

    hist = pd.DataFrame({"Cash": cash, "Valor acciones": valor_posiciones.sum(axis=1),
                         "Dividendos diarios": dividendos_diarios, "Valor cartera": valor_cartera,
                         "Rentabilidad diaria": valor_cartera.pct_change().fillna(0.0)})
    pesos = valor_posiciones.div(valor_cartera.replace(0, np.nan), axis=0).fillna(0.0)
    retornos_activos = ((precios + dividendos) / precios.shift(1) - 1).dropna(how="all")
    return {"hist": hist, "posiciones": posiciones, "pesos": pesos, "precios": precios,
            "dividendos": dividendos, "retornos_activos": retornos_activos, "operaciones": ops}


def historico_valor_cartera(archivo, fecha_fin=None, capital_inicial=10_000_000, hoja="Operativa",
                            incluir_costes=True, universo_tickers=(), lookback_dias=0):
    return _reconstruir_cartera_diaria(archivo, fecha_fin, capital_inicial, hoja, incluir_costes,
                                       universo_tickers, lookback_dias)["hist"]


def historico_posiciones_cartera(archivo, fecha_fin=None, capital_inicial=10_000_000, hoja="Operativa",
                                 incluir_costes=True, universo_tickers=(), lookback_dias=0):
    datos = _reconstruir_cartera_diaria(archivo, fecha_fin, capital_inicial, hoja, incluir_costes,
                                        universo_tickers, lookback_dias)
    return datos["hist"], datos["posiciones"], datos["pesos"], datos["precios"], datos["retornos_activos"]


def pnl_por_activo(archivo, fecha_fin=None, capital_inicial=10_000_000, hoja="Operativa",
                   incluir_costes=True, nombres_cortos=None, formato="presentacion"):
    nombres_cortos = nombres_cortos or {
        "SAN.MC": "B. Santander", "BBVA.MC": "BBVA", "BNP.PA": "BNP", "INGA.AS": "ING",
        "ADS.DE": "Adidas", "ASML.AS": "ASML", "IFX.DE": "Infineon", "ENR.DE": "Siemens Energy",
        "UCG.MI": "UniCredit", "SAF.PA": "Safran", "SGO.PA": "Saint-Gobain", "RACE.MI": "Ferrari",
        "RMS.PA": "Hermès", "ARGX.BR": "Argenx", "ENI.MI": "ENI", "ITX.MC": "Inditex",
        "AIR.PA": "Airbus", "ADYEN.AS": "Adyen", "EL.PA": "EssilorLuxottica",
        "WKL.AS": "Wolters Kluwer", "PRX.AS": "Prosus", "SAP.DE": "SAP", "BAYN.DE": "Bayer",
        "TTE.PA": "TotalEnergies", "RHM.DE": "Rheinmetall", "DBK.DE": "Deutsche Bank",
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
        posiciones.loc[r.fecha_trade:, r.ticker] += (1 if r.accion == "COMPRA" else -1) * abs(float(r.cantidad))

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

    datos = tabla.drop("TOTAL")
    datos = datos[~datos.index.to_series().str.contains("overn|swap|rate", case=False, regex=True)]
    print(f"P&L medio por activo: {float(datos['P&L (€)'].mean()):,.2f} €")

    tema = TEMA_MEMORIA if formato == "memoria" else TEMA_PRESENTACION
    texto = tema.get("texto", tema["blanco"])
    total_color = tema.get("destacado", tema.get("naranja", tema["lineas"]))
    colores = np.where(datos["P&L (€)"] >= 0, tema["positivo"], tema["negativo"])

    fig, ax = plt.subplots(figsize=(12.5, max(5, 0.32 * len(datos))), dpi=180)
    fig.patch.set_facecolor(tema["fondo"])
    ax.set_facecolor(tema["fondo"])

    ax.barh(datos.index, datos["P&L (€)"], color=colores, alpha=0.92)
    ax.axvline(0, color=texto, linewidth=1, alpha=0.55)
    ax.axvline(total, color=total_color, linewidth=2.2, linestyle="--",
               label=f"P&L total: {total:,.0f} €")

    ax.set_title("P&L por activo", color=texto, fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("P&L neto (€)", color=texto, fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f} €"))
    _estilizar_ejes(ax, tema)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=14)
    _estilizar_leyenda(leg, tema)

    plt.tight_layout()
    plt.show()
    return tabla, fig, ax


def tabla_semanal_atribucion(archivo, universo_tickers, fecha_fin=None, capital_inicial=10_000_000,
                             hoja="Operativa", incluir_costes=True, pesos_bmk=None, benchmark="^STOXX50E"):
    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin, universo_tickers)
    pesos_bmk = calcular_pesos_bmk_actuales(universo_tickers) if pesos_bmk is None else pesos_bmk
    pesos_bmk = pd.Series(pesos_bmk, dtype=float)
    pesos_bmk = pesos_bmk[pesos_bmk > 0] / pesos_bmk[pesos_bmk > 0].sum()

    fecha_valor = precios.index[precios.index <= fecha_fin].max()
    fechas_op = sorted(f for f in ops["fecha_trade"].unique() if f <= fecha_valor)
    if not fechas_op:
        raise ValueError("No hay fechas de operación válidas para la atribución.")

    fechas_bmk = sorted(set(fechas_op + [fecha_valor]))
    serie_bmk = _serie_benchmark(benchmark, fechas_bmk, min(fechas_bmk), max(fechas_bmk))
    pos, cash, filas = {}, float(capital_inicial), []

    def valor_posicion(fecha):
        return sum(q * precios.loc[fecha, t] for t, q in pos.items()
                   if t in precios.columns and pd.notna(precios.loc[fecha, t]))

    def retorno_total(t, f0, f1, divs):
        p0, p1 = precios.loc[f0, t], precios.loc[f1, t]
        return np.nan if pd.isna(p0) or pd.isna(p1) or p0 == 0 else (p1 + divs[t].sum()) / p0 - 1

    for i, f0 in enumerate(fechas_op):
        f1 = fechas_op[i + 1] if i + 1 < len(fechas_op) else fecha_valor
        nav0 = cash + valor_posicion(f0)
        if nav0 <= 0 or pd.isna(nav0) or f0 == f1:
            continue

        cash_sin_costes = cash
        for r in ops[ops["fecha_trade"].eq(f0)].itertuples():
            signo, cantidad = (1 if r.accion == "COMPRA" else -1), abs(float(r.cantidad))
            pos[r.ticker] = pos.get(r.ticker, 0.0) + signo * cantidad
            cash_sin_costes -= signo * cantidad * float(r.precio)
            cash -= signo * cantidad * float(r.precio_ejecutado)
        pos = {t: q for t, q in pos.items() if abs(q) > 1e-12}

        coste_eur = cash_sin_costes - cash
        divs = dividendos.loc[(dividendos.index > f0) & (dividendos.index <= f1)]
        r_bmk = float(serie_bmk.loc[f1] / serie_bmk.loc[f0] - 1)

        ret_universo = pd.Series({t: retorno_total(t, f0, f1, divs) for t in universo_tickers if t in precios.columns}).dropna()
        pesos_periodo = pesos_bmk.reindex(ret_universo.index).dropna()
        pesos_periodo = pesos_periodo / pesos_periodo.sum() if len(pesos_periodo) else pd.Series(dtype=float)

        seleccion = pesos = 0.0
        for t, q in pos.items():
            if t not in precios.columns:
                continue
            r, p0 = retorno_total(t, f0, f1, divs), precios.loc[f0, t]
            if pd.isna(r) or pd.isna(p0):
                continue
            w, b, exceso = q * p0 / nav0, float(pesos_periodo.get(t, 0.0)), r - r_bmk
            seleccion += b * exceso
            pesos += (w - b) * exceso

        coste = -coste_eur / nav0
        pesos += (cash_sin_costes / nav0) * (0.0 - r_bmk)
        divs_eur = sum(q * divs[t].sum() for t, q in pos.items() if t in divs.columns)
        nav1 = cash + divs_eur + valor_posicion(f1)
        r_cartera = nav1 / nav0 - 1

        filas.append({"Periodo": f"{pd.Timestamp(f0).date()} → {pd.Timestamp(f1).date()}",
                      "NAV inicial": nav0, "NAV final": nav1, "Rent. cartera neta": r_cartera,
                      "Rent. BMK": r_bmk, "Alpha": r_cartera - r_bmk, "Ef. selección": seleccion,
                      "Ef. pesos": pesos, "Costes": coste, "Costes (€)": -coste_eur,
                      "Dividendos (€)": divs_eur, "Check": r_cartera - (r_bmk + seleccion + pesos + coste)})
        cash += divs_eur

    return pd.DataFrame(filas).set_index("Periodo")


def encadenar_atribucion(semanal, capital_inicial=10_000_000):
    nav, bmk = float(capital_inicial), float(capital_inicial)
    comp = {"Ef. selección (€)": 0.0, "Ef. pesos (€)": 0.0, "Costes (€)": 0.0}
    filas = []

    for periodo, row in semanal.iterrows():
        r_cartera, r_bmk, bmk_ini = float(row["Rent. cartera neta"]), float(row["Rent. BMK"]), bmk
        efectos = {"Ef. selección (€)": float(row["Ef. selección"]),
                   "Ef. pesos (€)": float(row["Ef. pesos"]), "Costes (€)": float(row["Costes"])}
        for nombre, efecto in efectos.items():
            comp[nombre] = comp[nombre] * (1 + r_cartera) + bmk_ini * efecto

        nav *= 1 + r_cartera
        bmk *= 1 + r_bmk
        filas.append({"Periodo": periodo, "NAV cartera": nav, "NAV BMK": bmk,
                      "Alpha acumulado (€)": nav - bmk, **comp,
                      "Check alpha (€)": sum(comp.values()) - (nav - bmk)})

    final = pd.Series({"NAV cartera": nav, "NAV BMK": bmk, "Rentabilidad cartera": nav / capital_inicial - 1,
                       "Rentabilidad BMK": bmk / capital_inicial - 1, "Resultado cartera (€)": nav - capital_inicial,
                       "Efecto mercado (€)": bmk - capital_inicial, "Alpha (€)": nav - bmk, **comp})
    final["Check alpha (€)"] = final["Ef. selección (€)"] + final["Ef. pesos (€)"] + final["Costes (€)"] - final["Alpha (€)"]
    final["Check NAV (€)"] = final["Efecto mercado (€)"] + final["Ef. selección (€)"] + final["Ef. pesos (€)"] + final["Costes (€)"] - final["Resultado cartera (€)"]
    return pd.DataFrame(filas).set_index("Periodo"), final


def _pct(x): return "" if pd.isna(x) else f"{float(x):.2%}"
def _keur(x): return "" if pd.isna(x) else f"{float(x) / 1000:,.1f} k€"
def _num(x): return "" if pd.isna(x) else f"{float(x):.2f}"
def _vol_anual(r, p=52): return r.std(ddof=1) * np.sqrt(p)
def _sharpe(r, p=52):
    v = r.std(ddof=1)
    return np.nan if v == 0 or pd.isna(v) else r.mean() / v * np.sqrt(p)


def formatear_atribucion(semanal, final):
    cols = ["NAV inicial", "NAV final", "Rent. cartera neta", "Alpha", "Ef. selección", "Ef. pesos", "Costes"]
    semanal_fmt = semanal[[c for c in cols if c in semanal.columns]].copy().astype(object)
    for c in ["Rent. cartera neta", "Alpha", "Ef. selección", "Ef. pesos", "Costes"]:
        if c in semanal_fmt:
            semanal_fmt[c] = semanal[c].map(_pct)
    for c in ["NAV inicial", "NAV final"]:
        if c in semanal_fmt:
            semanal_fmt[c] = semanal[c].map(_keur)

    r_cartera, r_bmk = semanal["Rent. cartera neta"], semanal["Rent. BMK"]
    capital_ini = final["NAV cartera"] - final["Resultado cartera (€)"]
    final_fmt = pd.DataFrame({
        "Cartera": {"NAV final": _keur(final["NAV cartera"]),
                    "Resultado": f"{_keur(final['Resultado cartera (€)'])} ({_pct(final['Rentabilidad cartera'])})",
                    "Rentabilidad": _pct(final["Rentabilidad cartera"]),
                    "Volatilidad anualizada": _pct(_vol_anual(r_cartera)), "Sharpe": _num(_sharpe(r_cartera)),
                    "Alpha vs BMK": f"{_keur(final['Alpha (€)'])} ({_pct(final['Alpha (€)'] / capital_ini)})",
                    "Ef. selección": f"{_keur(final['Ef. selección (€)'])} ({_pct(final['Ef. selección (€)'] / capital_ini)})",
                    "Ef. pesos": f"{_keur(final['Ef. pesos (€)'])} ({_pct(final['Ef. pesos (€)'] / capital_ini)})",
                    "Costes": f"{_keur(final['Costes (€)'])} ({_pct(final['Costes (€)'] / capital_ini)})"},
        "BMK": {"NAV final": _keur(final["NAV BMK"]),
                "Resultado": f"{_keur(final['Efecto mercado (€)'])} ({_pct(final['Rentabilidad BMK'])})",
                "Rentabilidad": _pct(final["Rentabilidad BMK"]),
                "Volatilidad anualizada": _pct(_vol_anual(r_bmk)), "Sharpe": _num(_sharpe(r_bmk)),
                "Alpha vs BMK": "", "Ef. selección": "", "Ef. pesos": "", "Costes": ""},
    })
    return semanal_fmt, final_fmt


def _fecha_fin_real(semanal):
    return pd.to_datetime(str(semanal.index[-1]).split(" → ")[1])


def _series_y_metricas(archivo, semanal, final, fecha_fin=None, capital_inicial=10_000_000,
                       hoja="Operativa", incluir_costes=True, benchmark="^STOXX50E", rf_anual=0.02):
    fecha_fin_real = _fecha_fin_real(semanal)
    hist = historico_valor_cartera(archivo, fecha_fin, capital_inicial, hoja, incluir_costes)
    cartera = hist.loc[:fecha_fin_real, "Valor cartera"].copy()
    cartera.name = "Estrategia"

    datos_bmk = yf.download(benchmark, start=cartera.index.min().strftime("%Y-%m-%d"),
                            end=(fecha_fin_real + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                            auto_adjust=True, progress=False)
    bmk = datos_bmk["Close"].squeeze().reindex(cartera.index).ffill()
    bmk = capital_inicial * bmk / bmk.iloc[0]
    bmk.name = "STOXX 50"

    series = pd.concat([cartera, bmk], axis=1).dropna(how="all")
    if abs(series["Estrategia"].iloc[-1] - final["NAV cartera"]) > 1:
        raise ValueError("La cartera diaria no cuadra con el NAV final de la atribución.")

    retornos = series.pct_change().dropna()
    exceso = retornos - ((1 + rf_anual) ** (1 / 252) - 1)
    tabla_metricas = pd.DataFrame(index=series.columns)
    tabla_metricas["Rentabilidad"] = [series[c].iloc[-1] / capital_inicial - 1 for c in series.columns]
    tabla_metricas["Volatilidad"] = retornos.std() * np.sqrt(252)
    tabla_metricas["Max DD"] = (series / series.cummax() - 1).min()
    tabla_metricas["Sharpe"] = (exceso.mean() / retornos.std()) * np.sqrt(252)
    tabla_metricas = tabla_metricas.reset_index().rename(columns={"index": "Estrategia"})
    return series, tabla_metricas, estilo_tabla_metricas(tabla_metricas)


def resultados_cartera_bmk(archivo, universo_tickers, fecha_fin=None, capital_inicial=10_000_000,
                           hoja="Operativa", incluir_costes=True, pesos_bmk=None,
                           benchmark="^STOXX50E", rf_anual=0.02):
    semanal = tabla_semanal_atribucion(archivo, universo_tickers, fecha_fin, capital_inicial,
                                       hoja, incluir_costes, pesos_bmk, benchmark)
    detalle_acumulado, final = encadenar_atribucion(semanal, capital_inicial)
    semanal_fmt, final_fmt = formatear_atribucion(semanal, final)
    series, tabla_metricas, tabla_metricas_fmt = _series_y_metricas(
        archivo, semanal, final, fecha_fin, capital_inicial, hoja, incluir_costes, benchmark, rf_anual)
    return {"series": series, "semanal": semanal, "detalle_acumulado": detalle_acumulado, "final": final,
            "semanal_fmt": semanal_fmt, "final_fmt": final_fmt,
            "tabla_metricas": tabla_metricas, "tabla_metricas_fmt": tabla_metricas_fmt}


def series_diarias_cartera_bmks(archivo, universo_tickers, fecha_fin=None, capital_inicial=10_000_000,
                                hoja="Operativa", incluir_costes=True, benchmark="^STOXX50E", rf_anual=0.02):
    res = resultados_cartera_bmk(archivo, universo_tickers, fecha_fin, capital_inicial,
                                 hoja, incluir_costes, None, benchmark, rf_anual)
    return res["series"], res["semanal"], res["final"], res["tabla_metricas"], res["tabla_metricas_fmt"]


def estilo_tabla_metricas(tabla_metricas):
    estilos = [{"selector": "th", "props": [("background-color", "#28669A"), ("color", "white"),
                                            ("font-weight", "bold"), ("text-align", "center"),
                                            ("border", "1px solid #8AA6C1")]},
               {"selector": "td", "props": [("background-color", "white"), ("color", "black"),
                                            ("font-weight", "bold"), ("text-align", "center"),
                                            ("border", "1px solid #E0E0E0"), ("padding", "10px")]}]
    return tabla_metricas.style.format({"Rentabilidad": "{:.2%}", "Volatilidad": "{:.2%}",
                                        "Max DD": "{:.2%}", "Sharpe": "{:.2f}"}).set_table_styles(estilos).hide(axis="index")


def tabla_mpl_presentacion(df, titulo=None, figsize=(6.5, 2.8), dpi=220, bbox=(0.04, 0.05, 0.92, 0.76),
                           anchos=None, columnas_signo=None, tema=None, fontsize=9.5):
    tema = {**TEMA_PRESENTACION, **(tema or {})}
    df = df.copy().astype(str)
    columnas_signo = set(df.columns[1:] if columnas_signo is None else columnas_signo)

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
            continue

        valor = str(df.iloc[fila - 1, col])
        color = tema["positivo"] if df.columns[col] in columnas_signo and valor.startswith("+") else tema["negativo"] if df.columns[col] in columnas_signo and valor.startswith("-") else tema["blanco"]
        celda.set_facecolor(tema["panel"])
        celda.get_text().set_color(color)
        celda.get_text().set_weight("bold")

    if titulo:
        ax.text(0.5, 0.88, titulo, ha="center", va="bottom", color=tema["blanco"],
                fontsize=14, fontweight="bold", transform=ax.transAxes)
    plt.tight_layout(pad=0.25)
    plt.show()
    return fig, ax


def tabla_semanal_presentacion(semanal, ultimas=10, titulo="Rentabilidades semanales"):
    cols = {"Rent. cartera neta": "Cartera", "Rent. BMK": "BMK", "Alpha": "Alpha",
            "Ef. selección": "Selección", "Ef. pesos": "Pesos", "Costes": "Costes"}
    df = semanal[list(cols)].rename(columns=cols).tail(ultimas).copy()
    df.insert(0, "Semana", [str(i + 1) for i in range(len(df))])
    for c in df.columns[1:]:
        df[c] = df[c].map(lambda x: f"{float(x) * 100:+.2f}%")
    anchos = {0: 0.10, 1: 0.13, 2: 0.13, 3: 0.13, 4: 0.16, 5: 0.13, 6: 0.12}
    return tabla_mpl_presentacion(df, titulo=titulo, figsize=(8.6, 4.4), bbox=(0.04, 0.04, 0.92, 0.80),
                                  anchos=anchos, columnas_signo=df.columns[1:], fontsize=9.2)


def tabla_metricas_presentacion(tabla_metricas, metricas=("Rentabilidad", "Volatilidad", "Sharpe"),
                                titulo="Métricas finales", transpuesta=True):
    df = tabla_metricas.set_index("Estrategia")[list(metricas)].copy().astype(object)
    pct_cols = {"rentabilidad", "volatilidad", "vola", "max dd", "max drawdown"}

    if transpuesta:
        df = df.T.rename(index={"Volatilidad": "Vola"}).astype(object)
        for fila in df.index:
            df.loc[fila] = df.loc[fila].map(lambda x: f"{float(x):.2%}" if fila.lower() in pct_cols else f"{float(x):.2f}")
        df = df.reset_index().rename(columns={"index": "Métrica"})
        figsize, anchos, bbox = (4.4 + 0.8 * max(0, df.shape[1] - 3), 2.1 + 0.25 * max(0, df.shape[0] - 3)), {0: 0.30, **{i: 0.32 for i in range(1, df.shape[1])}}, (0.03, 0.08, 0.94, 0.72)
    else:
        for c in df.columns:
            df[c] = df[c].map(lambda x: f"{float(x):.2%}" if c.lower() in pct_cols else f"{float(x):.2f}")
        df = df.reset_index()
        figsize, anchos, bbox = (6.8, 2.6 + 0.25 * max(0, len(df) - 2)), None, (0.04, 0.06, 0.92, 0.72)

    return tabla_mpl_presentacion(df, titulo=titulo, figsize=figsize, bbox=bbox,
                                  anchos=anchos, columnas_signo=[], fontsize=9.6)


def _estilizar_ejes(ax, tema=None):
    tema = {**TEMA_PRESENTACION, **(tema or {})}
    texto = tema.get("texto", tema["blanco"])
    grid = tema.get("grid", tema["blanco"])
    alpha_grid = 0.75 if tema["fondo"] == "#FFFFFF" else 0.12

    ax.set_axisbelow(True)
    ax.tick_params(colors=texto, labelsize=11)
    ax.grid(True, color=grid, alpha=alpha_grid, linewidth=0.8, zorder=0)

    for s in ax.spines.values():
        s.set_color(tema["lineas"])


def _estilizar_leyenda(leg, tema=None):
    tema = {**TEMA_PRESENTACION, **(tema or {})}
    texto = tema.get("texto", tema["blanco"])

    leg.get_frame().set_facecolor(tema["panel"])
    leg.get_frame().set_edgecolor(tema["lineas"])
    leg.get_frame().set_alpha(0.92)
    for t in leg.get_texts():
        t.set_color(texto)


def grafico_evolucion_drawdown(series, titulo="Evolución de la cartera y drawdown", tema=None):
    tema = {**TEMA_PRESENTACION, **(tema or {})}
    texto = tema.get("texto", tema["blanco"])

    cols = [c for c in ["Estrategia", "STOXX 50"] if c in series.columns]
    dd = series[cols].div(series[cols].cummax()).sub(1)
    colores = {"Estrategia": tema["azul"], "STOXX 50": tema["naranja"]}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08}
    )
    fig.patch.set_facecolor(tema["fondo"])

    for ax in (ax1, ax2):
        ax.set_facecolor(tema["fondo"])
        _estilizar_ejes(ax, tema)

    for col in cols:
        ax1.plot(series.index, series[col], color=colores[col], linewidth=2.7, label=col)
        ax2.plot(dd.index, dd[col], color=colores[col], linewidth=2.0, label=f"{col} ({dd[col].min():.2%})")

    if "Estrategia" in dd:
        ax2.fill_between(dd.index, dd["Estrategia"], 0, color=tema["negativo"], alpha=0.18)

    ax1.axhline(series.iloc[0, 0], color=tema["lineas"], linestyle="--", linewidth=1, alpha=0.75)
    ax2.axhline(0, color=tema["lineas"], linewidth=1, alpha=0.9)

    ax1.set_title(titulo, color=texto, fontsize=22, fontweight="bold", pad=16)
    ax1.set_ylabel("Valor cartera", color=texto, fontsize=13)
    ax2.set_ylabel("Drawdown", color=texto, fontsize=13)
    ax2.set_xlabel("Fecha", color=texto, fontsize=13)

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.2f} M€"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.set_ylim(dd.min().min() * 1.15, 0.003)

    for ax, loc, fs in [(ax1, "upper left", 11), (ax2, "lower left", 10)]:
        leg = ax.legend(loc=loc, frameon=True, fontsize=fs)
        _estilizar_leyenda(leg, tema)

    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)


def tabla_descomposicion_efectos(final):
    capital_ini = final["NAV cartera"] - final["Resultado cartera (€)"]
    df = pd.DataFrame({
        "Impacto (€)": pd.Series({
            "Efecto mercado": final["Efecto mercado (€)"],
            "Ef. selección": final["Ef. selección (€)"],
            "Ef. pesos": final["Ef. pesos (€)"],
            "Costes": final["Costes (€)"],
            "Resultado total": final["Resultado cartera (€)"],
        })
    })
    df["Impacto (%)"] = df["Impacto (€)"] / capital_ini
    return df


def grafico_descomposicion_efectos(final, formato="memoria",
                                   titulo="Descomposición del resultado de la cartera"):
    tema = TEMA_MEMORIA if formato == "memoria" else TEMA_PRESENTACION
    texto = tema.get("texto", tema["blanco"])
    total_color = tema.get("destacado", tema.get("naranja", tema["azul"]))

    df = tabla_descomposicion_efectos(final)
    efectos = df.iloc[:-1].copy()
    total = df.loc["Resultado total", "Impacto (€)"]

    vals = efectos["Impacto (€)"].to_numpy()
    prev = np.r_[0, np.cumsum(vals)[:-1]]
    post = np.cumsum(vals)

    bottoms = np.minimum(prev, post)
    heights = np.abs(vals)
    colores = [tema["positivo"] if v >= 0 else tema["negativo"] for v in vals]

    x = np.arange(len(vals))
    x_total = len(vals) + 0.9

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=180)
    fig.patch.set_facecolor(tema["fondo"])
    ax.set_facecolor(tema["fondo"])
    ax.set_axisbelow(True)

    ax.bar(x, heights, bottom=bottoms, color=colores, width=0.62, zorder=3)
    ax.bar(x_total, abs(total), bottom=min(0, total), color=total_color, width=0.62, zorder=3)

    for i in range(len(vals) - 1):
        ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [post[i], post[i]],
                color=tema["lineas"], linewidth=1.5, zorder=4)

    ax.plot([x[-1] + 0.31, x_total - 0.31], [post[-1], post[-1]],
            color=tema["lineas"], linewidth=1.5, zorder=4)

    ymax_abs = max(abs(np.r_[0, prev, post, total]))
    offset = max(ymax_abs * 0.09, 22_000)

    for i, v in enumerate(vals):
        y = post[i] + (offset if v >= 0 else -offset)
        ax.text(
            x[i], y, f"{v/1000:+,.0f} k€",
            ha="center",
            va="bottom" if v >= 0 else "top",
            color=texto,
            fontsize=15,
            fontweight="bold"
        )

    ax.text(
        x_total, total + (offset if total >= 0 else -offset),
        f"{total/1000:+,.0f} k€",
        ha="center",
        va="bottom" if total >= 0 else "top",
        color=texto,
        fontsize=15,
        fontweight="bold"
    )

    ax.axhline(0, color=texto, linewidth=1.3, alpha=0.55, zorder=2)

    ax.set_title(titulo, color=texto, fontsize=28, fontweight="bold", pad=18)
    ax.set_ylabel("Impacto acumulado (€)", color=texto, fontsize=18)
    ax.set_xticks(list(x) + [x_total])
    ax.set_xticklabels(
        ["Ef. mercado", "Ef. selección", "Ef. pesos", "Ef. costes", "Ef. total"],
        fontsize=16
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:,.0f} k€"))

    ymin = min(0, bottoms.min(), post.min(), total) - 2.0 * offset
    ymax = max(0, (bottoms + heights).max(), post.max(), total) + 2.0 * offset
    ax.set_ylim(ymin, ymax)

    _estilizar_ejes(ax, tema)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.label.set_size(18)

    plt.tight_layout()
    plt.show()
    return df, fig, ax


def tabla_aciertos_modelo(archivo, universo_tickers, semanal, fecha_fin=None, hoja="Operativa",
                          incluir_costes=True, benchmark="^STOXX50E", top_n=15):
    ops, fecha_fin = _leer_operaciones(archivo, fecha_fin, hoja, incluir_costes)
    precios, dividendos, ops = _datos_mercado(ops, fecha_fin, universo_tickers)

    fecha_valor = precios.index[precios.index <= fecha_fin].max()
    fechas_op = sorted(f for f in ops["fecha_trade"].unique() if f <= fecha_valor)
    fechas = fechas_op + [fecha_valor]
    bmk = _serie_benchmark(benchmark, fechas, min(fechas), max(fechas))

    pos, filas = {}, []

    for i, f0 in enumerate(fechas_op):
        f1 = fechas_op[i + 1] if i + 1 < len(fechas_op) else fecha_valor

        for r in ops[ops["fecha_trade"].eq(f0)].itertuples():
            signo = 1 if r.accion == "COMPRA" else -1
            pos[r.ticker] = pos.get(r.ticker, 0) + signo * abs(float(r.cantidad))
        pos = {t: q for t, q in pos.items() if abs(q) > 1e-12}

        div = dividendos.loc[(dividendos.index > f0) & (dividendos.index <= f1)].sum()
        ret = ((precios.loc[f1] + div) / precios.loc[f0] - 1).reindex(universo_tickers).dropna()
        ret_sel = ret.reindex(pos.keys()).dropna()
        r_bmk = bmk.loc[f1] / bmk.loc[f0] - 1

        filas.append({
            "Semana": f"Semana {i + 1}",
            "Top 15": ret_sel.index.isin(ret.nlargest(top_n).index).sum(),
            "Baten BMK": (ret_sel > r_bmk).sum(),
            "Alpha cartera": semanal["Alpha"].iloc[i],
        })

    return pd.DataFrame(filas)

