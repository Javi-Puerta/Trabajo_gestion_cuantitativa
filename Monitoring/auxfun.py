import pandas as pd
import yfinance as yf
from IPython.display import display

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

def resultados_semana(fecha_ini, fecha_fin, df, universo_tickers):
    precios = yf.download(universo_tickers, start=fecha_ini - pd.Timedelta(days=7),
                          end=fecha_fin + pd.Timedelta(days=1), auto_adjust=False, progress=False)["Close"].ffill()
    stoxx = yf.download("^STOXX50E", start=fecha_ini - pd.Timedelta(days=7),
                        end=fecha_fin + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"].squeeze().ffill()

    ret_universo = pd.Series({
        t: precios[t].asof(fecha_fin) / precios[t].asof(fecha_ini) - 1
        for t in universo_tickers if t in precios.columns
    }).sort_values(ascending=False)
    ranking = ret_universo.rank(ascending=False).astype(int)
    ret_top15 = ret_universo.iloc[14]
    ret_stoxx = stoxx.asof(fecha_fin) / stoxx.asof(fecha_ini) - 1

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
            "Top 15?": "✓" if ranking[t] <= 15 else "✗"
        })

    df_resultado = pd.DataFrame(resultados).set_index("Ticker").sort_values("Ranking")

    # Fila agregada
    ret_cartera = (df_resultado["Peso"] * df_resultado["Retorno"]).sum()
    df_resultado.loc["TOTAL"] = {
        "Peso": df_resultado["Peso"].sum(),
        "Retorno": ret_cartera,
        "P&L (€)": df_resultado["P&L (€)"].sum(),
        "Ranking": "-",
        "Diff vs Top15": ret_cartera - ret_top15,
        "Top 15?": f"Alpha vs BMK: {ret_cartera - ret_stoxx:+.2%}"
    }

    hit_rate = (df_resultado.iloc[:-1]["Top 15?"] == "✓").mean()
    print(f"Periodo: {fecha_ini.date()} → {fecha_fin.date()} | Hit rate: {hit_rate:.0%} ({(df_resultado.iloc[:-1]['Top 15?'] == '✓').sum()}/15 en top 15) | BMK: {ret_stoxx:.2%}")

    return df_resultado


def resultados_todas_semanas(df, universo_tickers):
    fechas_op = sorted(df["fecha"].unique())
    for i in range(len(fechas_op) - 1):
        tabla = resultados_semana(fechas_op[i], fechas_op[i + 1], df, universo_tickers)
        display(tabla.style.format({
            "Peso": "{:.2%}", "Retorno": "{:.2%}", 
            "P&L": "{:.0f}", "Diff vs Top15": "{:.2%}"
        }))
        print()