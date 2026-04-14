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

def cartera(fecha, df):
    '''Devuelve un diccionario con los componentes de la cartera en una fecha dada.'''
    df_fecha = df[df["fecha"] <= fecha]
    cartera = {}
    for _, row in df_fecha.iterrows():
        ticker = row["Ticker"]
        if row["Accion"] == "COMPRA":
            cantidad = row["Cantidad"]
        else:
            cantidad = -row["Cantidad"]
        cartera[ticker] = cartera.get(ticker, 0) + cantidad
    return {ticker: cantidad for ticker, cantidad in cartera.items() if cantidad != 0}

def calcular_cash_diario(df, capital_inicial=10000000, incluir_costes=True):
    cash = capital_inicial
    historial_cash = {}
    
    for fecha, grupo in df.groupby("fecha"):
        for _, row in grupo.iterrows():
            precio = row["Precio_Ejecutado"] if incluir_costes else row["Precio"]
            if row["Accion"] == "COMPRA":
                cash -= row["Cantidad"] * precio
            else:
                cash += row["Cantidad"] * precio
        historial_cash[fecha] = cash
    
    return pd.Series(historial_cash)

def valor_cartera_diario(df, capital_inicial=10000000, incluir_costes=True):
    '''Calcula el valor diario de la cartera usando precios de cierre de yfinance.'''
    tickers = df["Ticker"].unique().tolist()
    start_date = df["fecha"].min()
    end_date = df["fecha"].max() + pd.Timedelta(days=30)
    fechas = df["fecha"].unique()
    
    precios = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)["Close"]
    precios = precios.ffill()
    
    cash_diario = calcular_cash_diario(df, capital_inicial, incluir_costes).reindex(precios.index).ffill()
    
    historial = {}
    for fecha in precios.index:
        cartera_fecha = cartera(fecha, df)
        valor = cash_diario.get(fecha, capital_inicial)
        for ticker, cantidad in cartera_fecha.items():
            valor += cantidad * precios.loc[fecha, ticker]
        historial[fecha] = valor
    
    return pd.Series(historial)

def rentabilidad_semanal_por_periodo(df, capital_inicial=10000000):
    '''Calcula la rentabilidad de la cartera y el STOXX50 entre cada fecha de operación.'''
    fechas_op = sorted(df["fecha"].unique())
    tickers = df["Ticker"].unique().tolist()
    
    # Descargamos precios sin ajustar para la cartera y con ajustar para el benchmark
    start = fechas_op[0]
    end = fechas_op[-1] + pd.Timedelta(days=7)
    precios = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Close"].ffill()
    stoxx = yf.download("^STOXX50E", start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze().ffill()

    resultados = []
    for i in range(len(fechas_op) - 1):
        fecha_ini = fechas_op[i]
        fecha_fin = fechas_op[i + 1]

        # Pesos de la cartera en fecha_ini (sin costes, usamos Precio no Precio_Ejecutado)
        cartera_ini = cartera(fecha_ini, df)  # {ticker: cantidad}
        valor_total = sum(cartera_ini[t] * precios[t].asof(fecha_ini) for t in cartera_ini if t in precios.columns)
        pesos = {t: (cartera_ini[t] * precios[t].asof(fecha_ini)) / valor_total for t in cartera_ini if t in precios.columns}

        ret_cartera = sum(
            pesos[t] * (precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1)
            for t in pesos if t in precios.columns
        )

        ret_stoxx = stoxx.asof(fecha_fin) / stoxx.asof(fecha_ini) - 1

        resultados.append({
            "Periodo": f"{fecha_ini.date()} → {fecha_fin.date()}",
            "Ret. Cartera": ret_cartera,
            "Ret. STOXX50": ret_stoxx,
            "Alpha": ret_cartera - ret_stoxx
        })

    return pd.DataFrame(resultados).set_index("Periodo")

def analisis_semana(fecha_ini, fecha_fin, df, universo_tickers):
    '''Combina resultados y atribución en una única tabla por semana.'''
    precios = yf.download(universo_tickers, start=fecha_ini - pd.Timedelta(days=7),
                          end=fecha_fin + pd.Timedelta(days=1), auto_adjust=False, progress=False)["Close"].ffill()
    stoxx = yf.download("^STOXX50E", start=fecha_ini - pd.Timedelta(days=7),
                        end=fecha_fin + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"].squeeze().ffill()

    ret_universo = pd.Series({
        t: precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
        for t in universo_tickers if t in precios.columns
    }).sort_values(ascending=False)
    ret_universo = ret_universo.dropna()
    ranking = ret_universo.rank(ascending=False).astype(int)
    ret_top15 = ret_universo.iloc[14] if len(ret_universo) > 14 else ret_universo.iloc[-1]
    ret_stoxx = stoxx.asof(fecha_fin) / stoxx.asof(fecha_ini) - 1
    ret_bmk_ew = ret_universo.mean()
    peso_bmk = 1 / len(universo_tickers)

    cartera_ini = cartera(fecha_ini, df)
    valor_total = sum(cartera_ini[t] * precios[t].asof(fecha_ini) for t in cartera_ini if t in precios.columns)

    resultados = []
    for t, cantidad in cartera_ini.items():
        if t not in ret_universo:
            continue
        precio_ini = precios[t].asof(fecha_ini)
        precio_fin = precios[t].asof(fecha_fin)
        peso = (cantidad * precio_ini) / valor_total
        retorno = ret_universo[t]
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

    ret_cartera = (df_resultado["Peso"] * df_resultado["Retorno"]).sum()
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

    hit_rate = (df_resultado.iloc[:-1]["Top 15?"] == "✓").mean()
    print(f"Periodo: {fecha_ini.date()} → {fecha_fin.date()} | Ret. Cartera: {ret_cartera:.2%} | Hit rate: {hit_rate:.0%} | BMK STOXX: {ret_stoxx:.2%} | BMK EW: {ret_bmk_ew:.2%}")
    return df_resultado


def analisis_todas_semanas(df, universo_tickers):
    fechas_op = sorted(df["fecha"].unique())
    for i in range(len(fechas_op) - 1):
        tabla = analisis_semana(fechas_op[i], fechas_op[i + 1], df, universo_tickers)
        display(tabla.style.format({
            "Peso": "{:.2%}", "Retorno": "{:.2%}", "P&L (€)": "{:,.0f}",
            "Diff vs Top15": "{:.2%}", "Ef. Selección": "{:.2%}", "Ef. Peso": "{:.2%}"
        }))
        print()
        
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
        tickers_predichos = tabla_semana.index[tabla_semana.index != "TOTAL"]
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
                          end=fecha_fin + pd.Timedelta(days=1), auto_adjust=False, progress=False)["Close"].ffill()

    ret_universo = pd.Series({
        t: precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
        for t in universo_tickers if t in precios.columns
    }).sort_values(ascending=False)

    cartera_ini = cartera(fecha_ini, df)
    valor_total = sum(cartera_ini[t] * precios[t].asof(fecha_ini) for t in cartera_ini if t in precios.columns)
    pesos = {t: (cartera_ini[t] * precios[t].asof(fecha_ini)) / valor_total for t in cartera_ini if t in precios.columns}

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
                          auto_adjust=False, progress=False)["Close"].ffill()
    
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
    precios = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Close"].ffill()

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
