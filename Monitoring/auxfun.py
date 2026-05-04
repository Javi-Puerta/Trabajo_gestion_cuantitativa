import pandas as pd
import yfinance as yf
import numpy as np
from IPython.display import display

import pickle
import sys
sys.path.append("../")
from ProveedorDatos import YFinanceProvider
from VariablesTransformation import FeatureEngineer
from UniversoActivos import UniversoActivosEstatico

def _descargar_precios(tickers, start, end, auto_adjust=True):
    tickers = list(dict.fromkeys(tickers))
    precios = yf.download(
        tickers, start=start, end=end,
        auto_adjust=auto_adjust, progress=False
    )["Close"].ffill()

    if isinstance(precios, pd.Series):
        precios = precios.to_frame(tickers[0])

    return precios


def _valor_posicion(fecha, pos, precios):
    return sum(
        q * precios[t].asof(fecha)
        for t, q in pos.items()
        if t in precios.columns and pd.notna(precios[t].asof(fecha))
    )


def estado_cartera(fecha, df, capital_inicial=10_000_000, incluir_costes=True):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    ops = df[df["fecha"] <= pd.Timestamp(fecha)].sort_values("fecha")

    pos, cash = {}, float(capital_inicial)

    for r in ops.itertuples():
        accion = str(r.Accion).upper()

        if accion == "MANTENER":
            continue
        if accion not in {"COMPRA", "VENTA"}:
            continue

        precio = r.Precio_Ejecutado if incluir_costes else r.Precio

        if accion == "COMPRA":
            pos[r.Ticker] = pos.get(r.Ticker, 0.0) + r.Cantidad
            cash -= r.Cantidad * precio
        else:
            pos[r.Ticker] = pos.get(r.Ticker, 0.0) - r.Cantidad
            cash += r.Cantidad * precio

    pos = {t: q for t, q in pos.items() if q != 0}
    return pos, cash


def _nav(fecha, pos, cash, precios):
    return cash + _valor_posicion(fecha, pos, precios)

def cartera(fecha, df):
    '''Devuelve un diccionario con los componentes de la cartera en una fecha dada.'''
    pos, _ = estado_cartera(
        fecha=fecha,
        df=df,
        capital_inicial=0,
        incluir_costes=False
    )
    return pos

def calcular_cash_diario(df, capital_inicial=10_000_000, incluir_costes=True):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    cash = float(capital_inicial)
    historial_cash = {}

    for fecha, grupo in df.sort_values("fecha").groupby("fecha"):
        for r in grupo.itertuples():
            accion = str(r.Accion).upper()

            if accion == "MANTENER":
                continue
            if accion not in {"COMPRA", "VENTA"}:
                continue

            precio = r.Precio_Ejecutado if incluir_costes else r.Precio

            if accion == "COMPRA":
                cash -= r.Cantidad * precio
            else:
                cash += r.Cantidad * precio

        historial_cash[fecha] = cash

    return pd.Series(historial_cash)

def valor_cartera_diario(df, capital_inicial=10_000_000, incluir_costes=True, auto_adjust=True):
    '''Calcula el valor diario de la cartera usando precios de cierre de yfinance.'''
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    tickers = df["Ticker"].unique().tolist()
    start = df["fecha"].min()
    end = df["fecha"].max() + pd.Timedelta(days=30)

    precios = _descargar_precios(tickers, start, end, auto_adjust=auto_adjust)

    historial = {}
    for fecha in precios.index:
        pos, cash = estado_cartera(fecha, df, capital_inicial, incluir_costes)
        historial[fecha] = _nav(fecha, pos, cash, precios)

    return pd.Series(historial)

def resumen_nav_desagregacion(df, universo_tickers, capital_inicial=10_000_000,
                              benchmark="^STOXX50E", fecha_valor=None,
                              auto_adjust=True):
    _, _, final = _calcular_atribucion(
        df, universo_tickers, capital_inicial, benchmark, fecha_valor, auto_adjust
    )
    return final

def rentabilidad_semanal_por_periodo(df, capital_inicial=10_000_000,
                                     incluir_costes=True, auto_adjust=True):
    '''Calcula la rentabilidad de la cartera y el STOXX50 entre cada fecha de operación.'''
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    fechas_op = sorted(df["fecha"].unique())
    tickers = df["Ticker"].unique().tolist()

    start = fechas_op[0] - pd.Timedelta(days=7)
    end = fechas_op[-1] + pd.Timedelta(days=7)

    precios = _descargar_precios(tickers, start, end, auto_adjust=auto_adjust)
    stoxx = yf.download(
        "^STOXX50E", start=start, end=end,
        auto_adjust=auto_adjust, progress=False
    )["Close"].squeeze().ffill()

    resultados = []

    for fecha_ini, fecha_fin in zip(fechas_op[:-1], fechas_op[1:]):
        pos, cash = estado_cartera(fecha_ini, df, capital_inicial, incluir_costes)

        nav_ini = _nav(fecha_ini, pos, cash, precios)
        nav_fin = _nav(fecha_fin, pos, cash, precios)

        if nav_ini == 0 or pd.isna(nav_ini) or pd.isna(nav_fin):
            continue

        ret_cartera = nav_fin / nav_ini - 1
        ret_stoxx = stoxx.asof(fecha_fin) / stoxx.asof(fecha_ini) - 1

        resultados.append({
            "Periodo": f"{fecha_ini.date()} → {fecha_fin.date()}",
            "Ret. Cartera": ret_cartera,
            "Ret. STOXX50": ret_stoxx,
            "Alpha": ret_cartera - ret_stoxx
        })

    return pd.DataFrame(resultados).set_index("Periodo")

def _calcular_atribucion(df, universo_tickers, capital_inicial=10_000_000,
                         benchmark="^STOXX50E", fecha_valor=None,
                         auto_adjust=True, guardar_detalle=False):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    fecha_valor = pd.Timestamp.today().normalize() if fecha_valor is None else pd.Timestamp(fecha_valor)

    fechas_op = sorted(df.loc[df["fecha"] <= fecha_valor, "fecha"].unique())
    tickers = sorted(set(df["Ticker"]) | set(universo_tickers))
    start, end = fechas_op[0] - pd.Timedelta(days=7), fecha_valor + pd.Timedelta(days=2)

    precios = yf.download(tickers, start=start, end=end,
                          auto_adjust=auto_adjust, progress=False)["Close"].ffill()
    if isinstance(precios, pd.Series):
        precios = precios.to_frame(tickers[0])

    bmk = yf.download(benchmark, start=start, end=end,
                      auto_adjust=auto_adjust, progress=False)["Close"].squeeze().ffill()

    fecha_valor = precios.index[precios.index <= fecha_valor].max()
    w_bmk = 1 / len(universo_tickers)

    pos, cash = {}, capital_inicial
    resumen, detalles = [], {}

    def valor_pos(fecha):
        return sum(
            q * precios[t].asof(fecha)
            for t, q in pos.items()
            if t in precios.columns and pd.notna(precios[t].asof(fecha))
        )

    for i, f0 in enumerate(fechas_op):
        f1 = fechas_op[i + 1] if i + 1 < len(fechas_op) else fecha_valor

        ops = df[df["fecha"].eq(f0)]
        cash_sin_coste = cash
        coste_eur = (ops["Cantidad"] * (ops["Precio_Ejecutado"] - ops["Precio"]).abs()).sum()

        for r in ops.itertuples():
            if r.Accion == "MANTENER":
                continue

            signo = 1 if r.Accion == "COMPRA" else -1
            pos[r.Ticker] = pos.get(r.Ticker, 0) + signo * r.Cantidad

            if r.Accion == "COMPRA":
                cash_sin_coste -= r.Cantidad * r.Precio
                cash -= r.Cantidad * r.Precio_Ejecutado
            else:
                cash_sin_coste += r.Cantidad * r.Precio
                cash += r.Cantidad * r.Precio_Ejecutado

        pos = {t: q for t, q in pos.items() if q != 0}

        nav0 = cash + valor_pos(f0)

        if nav0 <= 0:
            continue

        r_bmk = 0 if f1 == f0 else bmk.asof(f1) / bmk.asof(f0) - 1
        coste = coste_eur / nav0
        seleccion = peso = ret_bruta = 0.0
        filas = []

        for t, q in pos.items():
            if t not in precios.columns:
                continue

            p0, p1 = precios[t].asof(f0), precios[t].asof(f1)
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                continue

            r = 0 if f1 == f0 else p1 / p0 - 1
            w = q * p0 / nav0
            exceso = r - r_bmk

            ef_sel = w_bmk * exceso
            ef_peso = (w - w_bmk) * exceso

            ret_bruta += w * r
            seleccion += ef_sel
            peso += ef_peso

            filas.append({
                "Ticker": t,
                "Peso": w,
                "Retorno": r,
                "P&L bruto (€)": q * (p1 - p0),
                "Ef. Selección": ef_sel,
                "Ef. Peso": ef_peso,
            })

        peso += (cash_sin_coste / nav0) * (-r_bmk)
        ret_neta = r_bmk + seleccion + peso - coste

        periodo = f"{pd.Timestamp(f0).date()} → {pd.Timestamp(f1).date()}"
        resumen.append({
            "Periodo": periodo,
            "NAV inicial": nav0,
            "Rent. cartera neta": ret_neta,
            "Efecto mercado": r_bmk,
            "Ef. selección": seleccion,
            "Ef. pesos": peso,
            "Costes": -coste,
            "Resultado neto (€)": nav0 * ret_neta,
            "Mercado (€)": nav0 * r_bmk,
            "Selección (€)": nav0 * seleccion,
            "Pesos (€)": nav0 * peso,
            "Costes (€)": -coste_eur,
            "Check suma": ret_neta - (r_bmk + seleccion + peso - coste),
        })

        if guardar_detalle and filas:
            detalles[periodo] = pd.DataFrame(filas).set_index("Ticker")

    semanal = pd.DataFrame(resumen).set_index("Periodo")

    nav = cash + valor_pos(fecha_valor)
    final = pd.Series({
        "Fecha valoración": fecha_valor.date(),
        "NAV": nav,
        "NAV normalizado": nav / capital_inicial,
        "Rentabilidad NAV": nav / capital_inicial - 1,
        "Efecto mercado (€)": semanal["Mercado (€)"].sum(),
        "Ef. selección (€)": semanal["Selección (€)"].sum(),
        "Ef. pesos (€)": semanal["Pesos (€)"].sum(),
        "Costes (€)": semanal["Costes (€)"].sum(),
        "Resultado NAV (€)": nav - capital_inicial,
    })

    final["Check NAV (€)"] = (
        final["Efecto mercado (€)"]
        + final["Ef. selección (€)"]
        + final["Ef. pesos (€)"]
        + final["Costes (€)"]
        - final["Resultado NAV (€)"]
    )

    return semanal, detalles, final

def analisis_semana(fecha_ini, fecha_fin, df, universo_tickers,
                    capital_inicial=10_000_000, incluir_costes=True,
                    auto_adjust=True):
    '''Combina resultados y atribución en una única tabla por semana.'''
    fecha_ini = pd.Timestamp(fecha_ini)
    fecha_fin = pd.Timestamp(fecha_fin)

    precios = _descargar_precios(
        universo_tickers,
        fecha_ini - pd.Timedelta(days=7),
        fecha_fin + pd.Timedelta(days=1),
        auto_adjust=auto_adjust
    )

    stoxx = yf.download(
        "^STOXX50E",
        start=fecha_ini - pd.Timedelta(days=7),
        end=fecha_fin + pd.Timedelta(days=1),
        auto_adjust=auto_adjust,
        progress=False
    )["Close"].squeeze().ffill()

    ret_universo = pd.Series({
        t: precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
        for t in universo_tickers
        if t in precios.columns
        and pd.notna(precios[t].asof(fecha_ini))
        and pd.notna(precios[t].asof(fecha_fin))
        and precios[t].asof(fecha_ini) != 0
    }).sort_values(ascending=False)

    ranking = ret_universo.rank(ascending=False).astype(int)
    ret_top15 = ret_universo.iloc[14] if len(ret_universo) > 14 else ret_universo.iloc[-1]
    ret_stoxx = stoxx.asof(fecha_fin) / stoxx.asof(fecha_ini) - 1
    ret_bmk_ew = ret_universo.mean()
    peso_bmk = 1 / len(universo_tickers)

    pos, cash = estado_cartera(fecha_ini, df, capital_inicial, incluir_costes)
    nav_ini = _nav(fecha_ini, pos, cash, precios)
    nav_fin = _nav(fecha_fin, pos, cash, precios)
    ret_cartera = nav_fin / nav_ini - 1

    resultados = []

    for t, cantidad in pos.items():
        if t not in ret_universo:
            continue

        precio_ini = precios[t].asof(fecha_ini)
        precio_fin = precios[t].asof(fecha_fin)
        retorno = ret_universo[t]
        peso = cantidad * precio_ini / nav_ini

        resultados.append({
            "Ticker": t,
            "Peso": peso,
            "Retorno": retorno,
            "P&L (€)": cantidad * (precio_fin - precio_ini),
            "Ranking": ranking[t],
            "Diff vs Top15": retorno - ret_top15,
            "Ef. Selección": peso_bmk * (retorno - ret_stoxx),
            "Ef. Peso": (peso - peso_bmk) * (retorno - ret_stoxx),
            "Top 15?": "✓" if ranking[t] <= 15 else "✗"
        })

    df_resultado = pd.DataFrame(resultados).set_index("Ticker").sort_values("Ranking")

    if abs(cash) > 1e-9:
        df_resultado.loc["CASH"] = {
            "Peso": cash / nav_ini,
            "Retorno": 0.0,
            "P&L (€)": 0.0,
            "Ranking": "-",
            "Diff vs Top15": -ret_top15,
            "Ef. Selección": 0.0,
            "Ef. Peso": (cash / nav_ini) * (-ret_stoxx),
            "Top 15?": "-"
        }

    df_resultado.loc["TOTAL"] = {
        "Peso": df_resultado["Peso"].sum(),
        "Retorno": ret_cartera,
        "P&L (€)": df_resultado["P&L (€)"].sum(),
        "Ranking": "-",
        "Diff vs Top15": ret_cartera - ret_top15,
        "Ef. Selección": df_resultado["Ef. Selección"].sum(),
        "Ef. Peso": df_resultado["Ef. Peso"].sum(),
        "Top 15?": f"Alpha vs STOXX: {ret_cartera - ret_stoxx:+.2%}"
    }

    tickers_reales = df_resultado.index.difference(["CASH", "TOTAL"])
    hit_rate = (df_resultado.loc[tickers_reales, "Top 15?"] == "✓").mean()

    print(
        f"Periodo: {fecha_ini.date()} → {fecha_fin.date()} | "
        f"Ret. Cartera: {ret_cartera:.2%} | "
        f"Hit rate: {hit_rate:.0%} | "
        f"BMK STOXX: {ret_stoxx:.2%} | "
        f"BMK EW: {ret_bmk_ew:.2%}"
    )

    return df_resultado


def analisis_todas_semanas(df, universo_tickers, capital_inicial=10_000_000,
                           benchmark="^STOXX50E", auto_adjust=True,
                           fecha_valor=None):
    semanal, _, _ = _calcular_atribucion(
        df=df,
        universo_tickers=universo_tickers,
        capital_inicial=capital_inicial,
        benchmark=benchmark,
        fecha_valor=fecha_valor,
        auto_adjust=auto_adjust,
        guardar_detalle=False,
    )

    tabla = semanal[[
        "Rent. cartera neta",
        "Efecto mercado",
        "Ef. selección",
        "Ef. pesos",
        "Costes",
    ]].copy()

    tabla["Alpha"] = (
        tabla["Ef. selección"]
        + tabla["Ef. pesos"]
        + tabla["Costes"]
    )

    tabla = tabla.rename(columns={
        "Rent. cartera neta": "Rent. cartera",
        "Efecto mercado": "Rent. benchmark",
        "Ef. selección": "Efecto selección",
        "Ef. pesos": "Efecto pesos",
        "Costes": "Efecto costes",
    })

    tabla = tabla[[
        "Rent. cartera",
        "Rent. benchmark",
        "Alpha",
        "Efecto selección",
        "Efecto pesos",
        "Efecto costes",
    ]]

    display(tabla.style.format({
        "Rent. cartera": "{:.2%}",
        "Rent. benchmark": "{:.2%}",
        "Alpha": "{:.2%}",
        "Efecto selección": "{:.2%}",
        "Efecto pesos": "{:.2%}",
        "Efecto costes": "{:.2%}",
    }))

    return tabla
        
def generar_tabla_aciertos(df, universo_tickers):
    """
    Genera una tabla con filas = empresas seleccionadas alguna vez,
    columnas = semanas, y celdas:
    - ✅ si la empresa fue predicha esa semana y estuvo en el top 15
    - ❌ si fue predicha pero no entró en el top 15
    - - si no fue predicha esa semana
    """
    fechas_op = sorted(df["fecha"].unique())
    semanas = []  # lista de etiquetas para las columnas
    datos = {}    # diccionario anidado: datos[ticker][semana] = símbolo

    for i in range(len(fechas_op) - 1):
        fecha_ini = fechas_op[i]
        fecha_fin = fechas_op[i + 1]
        etiqueta_semana = f"{fecha_ini.date()}→{fecha_fin.date()}"
        semanas.append(etiqueta_semana)

        # Obtener la tabla de análisis de la semana
        tabla_semana = analisis_semana(fecha_ini, fecha_fin, df, universo_tickers)

        # Extraer los tickers que fueron predichos (todas las filas excepto 'TOTAL')
        tickers_predichos = tabla_semana.index.difference(["TOTAL", "CASH"])
        # Mapear ticker -> si estuvo en top 15 (True/False)
        exito = {}
        for t in tickers_predichos:
            exito[t] = (tabla_semana.loc[t, "Top 15?"] == "✓")

        # Actualizar la matriz de datos
        for t in tickers_predichos:
            if t not in datos:
                datos[t] = {}
            datos[t][etiqueta_semana] = "✅" if exito[t] else "❌"

        # Para los tickers que ya habían aparecido en semanas anteriores pero no en esta,
        # se dejarán como '-' más adelante (rellenamos con NaN luego)

    # Construir DataFrame a partir del diccionario
    df_resumen = pd.DataFrame.from_dict(datos, orient='index')
    # Rellenar las celdas vacías (ticker no predicho esa semana) con '-'
    df_resumen = df_resumen.fillna('-')
    # Asegurar el orden de las columnas según las semanas
    df_resumen = df_resumen[semanas]
    # Ordenar filas alfabéticamente (opcional)
    df_resumen.sort_index(inplace=True)

    return df_resumen
        
def top_bottom_universo(fecha_ini, fecha_fin, df, universo_tickers, n=10):
    precios = yf.download(universo_tickers, start=fecha_ini - pd.Timedelta(days=7),
                          end=fecha_fin + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"].ffill()

    ret_universo = pd.Series({
        t: precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
        for t in universo_tickers if t in precios.columns
    }).sort_values(ascending=False)

    pos, cash = estado_cartera(fecha_ini, df, capital_inicial=10_000_000, incluir_costes=True)
    nav_ini = cash + _valor_posicion(fecha_ini, pos, precios)

    pesos = {
        t: q * precios[t].asof(fecha_ini) / nav_ini
        for t, q in pos.items()
        if t in precios.columns and pd.notna(precios[t].asof(fecha_ini))
    }

    def construir_tabla(serie):
        df_t = serie.reset_index()
        df_t.columns = ["Ticker", "Retorno"]
        df_t["Elegido?"] = df_t["Ticker"].apply(lambda t: "✓" if t in pesos else "✗")
        df_t["Peso"] = df_t["Ticker"].apply(lambda t: pesos.get(t, None))
        return df_t.set_index("Ticker")

    top = construir_tabla(ret_universo.head(n))
    bottom = construir_tabla(ret_universo.tail(n))

    print(f"Periodo: {fecha_ini.date()} → {fecha_fin.date()}")
    print(f"\n🔼 Top {n}")
    display(top.style.format({"Retorno": "{:.2%}", "Peso": lambda x: f"{x:.2%}" if x is not None else "-"}))
    print(f"\n🔽 Bottom {n}")
    display(bottom.style.format({"Retorno": "{:.2%}", "Peso": lambda x: f"{x:.2%}" if x is not None else "-"}))


def top_bottom_todas_semanas(df, universo_tickers, n=10):
    fechas_op = sorted(df["fecha"].unique())
    for i in range(len(fechas_op) - 1):
        top_bottom_universo(fechas_op[i], fechas_op[i + 1], df, universo_tickers, n)
        print()
        
def tabla_aciertos_semanales(df, universo_tickers):
    '''Muestra para cada semana: % de activos en Top15, Top30, Top45, Resto.'''
    fechas_op = sorted(df["fecha"].unique())
    precios = yf.download(universo_tickers, start=fechas_op[0] - pd.Timedelta(days=7),
                          end=fechas_op[-1] + pd.Timedelta(days=7), 
                          auto_adjust=True, progress=False)["Close"].ffill()
    
    resultados = []
    for i in range(len(fechas_op) - 1):
        fecha_ini, fecha_fin = fechas_op[i], fechas_op[i + 1]
        
        ret_universo = pd.Series({
            t: precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
            for t in universo_tickers if t in precios.columns
        })
        ranking = ret_universo.rank(ascending=False)
        
        cartera_ini = cartera(fecha_ini, df)
        tickers_elegidos = [t for t in cartera_ini.keys() if t in ranking]
        n_total = len(tickers_elegidos)
        
        if n_total == 0:
            continue
            
        ranks = ranking[tickers_elegidos]
        resultados.append({
            "Periodo": f"{fecha_ini.date()} → {fecha_fin.date()}",
            "Top 15": f"{(ranks <= 15).sum()}/{n_total}",
            "Top 30": f"{((ranks > 15) & (ranks <= 30)).sum()}/{n_total}",
            "Top 45": f"{((ranks > 30) & (ranks <= 45)).sum()}/{n_total}",
            "Bottom 5": f"{(ranks > 45).sum()}/{n_total}",
        })
    
    return pd.DataFrame(resultados)


def recuperar_scores_historicos(fechas_op, universo_tickers, modelo_path="../mi_cartera/modelo_estado.pkl"):
    '''Recupera los scores del modelo para cada fecha de operación usando el modelo guardado.'''
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    prov = YFinanceProvider()
    fe = FeatureEngineer(criterio=5, ticker_indice="^STOXX50E")
    universo = UniversoActivosEstatico(universo_tickers)

    start = fechas_op[0] - pd.DateOffset(years=5)
    end = fechas_op[-1] + pd.Timedelta(days=1)
    df_daily = prov.download_prices_daily(universo_tickers + ["^STOXX50E"], start, end)
    df_weekly = prov.download_prices_weekly(universo_tickers + ["^STOXX50E"], start, end)
    df = fe.build(df_weekly, df_daily)

    scores_por_fecha = {}
    for fecha in fechas_op:
        df_hoy = df[df["Fecha"] == pd.Timestamp(fecha)]
        if df_hoy.empty:
            continue
        proba = modelo.predict_proba(df_hoy[fe.feature_cols])
        scores_por_fecha[fecha] = dict(zip(df_hoy["Ticker"], proba))

    return scores_por_fecha


def analisis_scores(scores_por_fecha, df, universo_tickers, ultimas_n=None):
    '''Analiza la calibración del modelo comparando scores con rentabilidades reales.
    
    Parámetros:
    -----------
    ultimas_n : int, opcional
        Si se especifica, analiza solo las últimas N semanas
    '''
    fechas_op = sorted(scores_por_fecha.keys())
    
    # Filtrar solo las últimas N semanas si se especifica
    if ultimas_n is not None:
        fechas_op = fechas_op[-ultimas_n:]
        if len(fechas_op) == 0:
            print(f"No hay suficientes semanas. Total disponible: {len(sorted(scores_por_fecha.keys()))}")
            return pd.DataFrame()
    
    tickers = universo_tickers
    
    start = fechas_op[0] - pd.Timedelta(days=7)
    end = fechas_op[-1] + pd.Timedelta(days=7)
    precios = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"].ffill()

    filas = []
    for i in range(len(fechas_op) - 1):
        fecha_ini = fechas_op[i]
        fecha_fin = fechas_op[i + 1]
        scores = scores_por_fecha[fecha_ini]

        # Calculamos quintiles sobre todos los scores de la fecha
        scores_series = pd.Series(scores)
        quintiles = pd.qcut(scores_series, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

        for t, score in scores.items():
            if t not in precios.columns:
                continue
            retorno = precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
            elegido = t in cartera(fecha_ini, df)
            filas.append({
                "Fecha": fecha_ini,
                "Ticker": t,
                "Score": score,
                "Retorno": retorno,
                "Elegido": elegido,
                "Quintil": quintiles.get(t)
            })

    df_scores = pd.DataFrame(filas)

    # Rentabilidad media por quintil de score
    print(f"Rentabilidad media por quintil de score ({'todas las semanas' if ultimas_n is None else f'últimas {ultimas_n} semanas'}):")
    display(df_scores.groupby("Quintil")["Retorno"].mean().to_frame().style.format("{:.2%}"))

    # Score medio de elegidos vs no elegidos
    print("\nScore medio elegidos vs no elegidos:")
    display(df_scores.groupby("Elegido")["Score"].mean().to_frame().style.format("{:.3f}"))

    return df_scores

def resumen_por_activo(df, universo_tickers):
    """
    Genera una tabla resumen por activo con el desempeño agregado en todas las semanas.
    
    Parámetros:
    - df: DataFrame con columna 'fecha' (fechas de rebalanceo/inicio de semana)
    - universo_tickers: lista de tickers considerados
    
    Retorna:
    - DataFrame con filas = activos seleccionados alguna vez, columnas = métricas agregadas.
    """
    fechas_op = sorted(df["fecha"].unique())
    registros = []   # lista de dicts con una entrada por cada selección semanal

    for i in range(len(fechas_op) - 1):
        fecha_ini = fechas_op[i]
        fecha_fin = fechas_op[i + 1]
        tabla_semana = analisis_semana(fecha_ini, fecha_fin, df, universo_tickers)
        # Excluir la fila "TOTAL" que añade analisis_semana
        tabla_semana = tabla_semana[tabla_semana.index != "TOTAL"]
        
        for ticker, row in tabla_semana.iterrows():
            registros.append({
                "Ticker": ticker,
                "Seleccionado": 1,
                "Acierto": 1 if row["Top 15?"] == "✓" else 0,
                "Retorno": row["Retorno"],
                "Ranking": row["Ranking"] if row["Ranking"] != "-" else None
            })

    if not registros:
        print("No hay selecciones en el período.")
        return pd.DataFrame()

    df_reg = pd.DataFrame(registros)
    
    # Agregación por ticker
    resumen = df_reg.groupby("Ticker").agg(
        Veces_Seleccionado=("Seleccionado", "sum"),
        Aciertos=("Acierto", "sum"),
        Retorno_Promedio=("Retorno", "mean"),
        Ranking_Promedio=("Ranking", "mean")
    ).reset_index()
    
    # Calcular hit rate
    resumen["Hit_Rate"] = resumen["Aciertos"] / resumen["Veces_Seleccionado"]
    
    # Ordenar por hit rate descendente y después por frecuencia
    resumen = resumen.sort_values(["Hit_Rate", "Veces_Seleccionado"], ascending=[False, False])
    
    # Formateo para presentación
    resumen["Hit_Rate"] = resumen["Hit_Rate"].apply(lambda x: f"{x:.0%}")
    resumen["Retorno_Promedio"] = resumen["Retorno_Promedio"].apply(lambda x: f"{x:.2%}")
    resumen["Ranking_Promedio"] = resumen["Ranking_Promedio"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    )
    
    # Añadir fila de totales globales
    total_selecciones = df_reg["Seleccionado"].sum()
    total_aciertos = df_reg["Acierto"].sum()
    total_hit_rate = total_aciertos / total_selecciones if total_selecciones > 0 else 0
    total_ret_promedio = df_reg["Retorno"].mean()
    
    total_row = pd.DataFrame({
        "Ticker": ["TOTAL"],
        "Veces_Seleccionado": [total_selecciones],
        "Aciertos": [total_aciertos],
        "Hit_Rate": [f"{total_hit_rate:.0%}"],
        "Retorno_Promedio": [f"{total_ret_promedio:.2%}"],
        "Ranking_Promedio": ["-"]
    })
    
    resumen = pd.concat([resumen, total_row], ignore_index=True)
    return resumen

def grafico_score_vs_retorno(df_scores):
    """
    Scatter plot: Score del modelo vs Retorno real siguiente semana.
    Cada punto = un activo en una semana.
    Línea de regresión muestra la correlación.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    # Eliminar NaNs
    data = df_scores[["Score", "Retorno"]].dropna()
    
    if data.empty:
        print("No hay datos válidos para graficar")
        return
    
    # Calcular correlación
    corr, p_value = pearsonr(data["Score"], data["Retorno"])
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(data["Score"], data["Retorno"], alpha=0.4, s=20, color="steelblue")
    
    # Línea de regresión
    z = np.polyfit(data["Score"], data["Retorno"], 1)
    p = np.poly1d(z)
    ax.plot(data["Score"], p(data["Score"]), "r--", linewidth=2, 
            label=f"Regresión (pendiente={z[0]:.3f})")
    
    # Línea horizontal en y=0
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    # Etiquetas
    ax.set_xlabel("Score del modelo", fontsize=12)
    ax.set_ylabel("Retorno semana siguiente", fontsize=12)
    ax.set_title(f"Score vs Retorno Real | Correlación = {corr:.3f} (p={p_value:.4f})", 
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Formato de porcentajes en eje Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas adicionales
    print(f"\n📊 Correlación Score-Retorno: {corr:.4f}")
    print(f"   Significancia (p-value): {p_value:.4f}")
    print(f"   Pendiente regresión: {z[0]:.4f}")
    print(f"\n   Interpretación:")
    if abs(corr) < 0.1:
        print("   ❌ Correlación casi nula → modelo no predice")
    elif corr < 0:
        print("   🔴 Correlación NEGATIVA → modelo invertido")
    elif corr < 0.2:
        print("   🟡 Correlación débil → modelo poco útil")
    elif corr < 0.4:
        print("   🟢 Correlación moderada → modelo funcional")
    else:
        print("   ✅ Correlación fuerte → buen modelo")
