import pandas as pd
import yfinance as yf

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