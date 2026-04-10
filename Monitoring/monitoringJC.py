import pandas as pd
import yfinance as yf

def analizar_cartera(csv_path):
    # 1. Cargar y estandarizar los datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return "Error: No se encontró el archivo CSV. Verifica la ruta."

    # Estandarizamos los nombres de las columnas a minúsculas y quitamos espacios
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Procesar el libro de órdenes (Tracking Realizado y Latente)
    cartera = {}

    for index, row in df.iterrows():
        # Referencias actualizadas a tus nombres de columnas
        ticker = row['ticker']
        accion = str(row['accion']).strip().lower()
        cantidad = float(row['cantidad'])
        precio_ejec = float(row['precio_ejecutado'])

        if ticker not in cartera:
            cartera[ticker] = {'cantidad': 0.0, 'coste_total': 0.0, 'pnl_realizado': 0.0}

        if accion == 'compra':
            cartera[ticker]['cantidad'] += cantidad
            cartera[ticker]['coste_total'] += cantidad * precio_ejec
            
        elif accion == 'venta':
            if cartera[ticker]['cantidad'] > 0:
                # Calculamos el coste medio (VWAP) antes de la venta
                coste_medio_actual = cartera[ticker]['coste_total'] / cartera[ticker]['cantidad']
                
                # Reducimos la posición y el coste proporcional
                cartera[ticker]['cantidad'] -= cantidad
                cartera[ticker]['coste_total'] -= cantidad * coste_medio_actual
                
                # Calculamos el PnL Realizado usando el PRECIO EJECUTADO de la venta
                cartera[ticker]['pnl_realizado'] += cantidad * (precio_ejec - coste_medio_actual)

    # 3. Descargar precios de mercado y calcular métricas finales
    print("Obteniendo datos de mercado en tiempo real...\n")
    resultados = []
    
    for ticker, datos in cartera.items():
        cantidad = datos['cantidad']
        pnl_realizado = datos['pnl_realizado']
        
        # Si la posición está completamente cerrada
        if cantidad <= 0.0001:
            resultados.append({
                'Ticker': ticker,
                'Acciones': 0.0,
                'Precio Medio': 0.0,
                'Precio Mercado': 0.0,
                'Valor Mercado': 0.0,
                'PnL Latente': 0.0,
                'PnL Realizado': round(pnl_realizado, 2),
                'PnL Total': round(pnl_realizado, 2)
            })
            continue

        coste_medio = datos['coste_total'] / cantidad
        
        try:
            stock = yf.Ticker(ticker)
            precio_mercado = stock.fast_info['lastPrice']
        except Exception as e:
            print(f"Aviso: No se pudo obtener el precio de {ticker}. Usando coste medio.")
            precio_mercado = coste_medio 

        valor_mercado = cantidad * precio_mercado
        pnl_latente = valor_mercado - datos['coste_total']
        pnl_total = pnl_latente + pnl_realizado

        resultados.append({
            'Ticker': ticker,
            'Acciones': cantidad,
            'Precio Medio': round(coste_medio, 2),
            'Precio Mercado': round(precio_mercado, 2),
            'Valor Mercado': round(valor_mercado, 2),
            'PnL Latente': round(pnl_latente, 2),
            'PnL Realizado': round(pnl_realizado, 2),
            'PnL Total': round(pnl_total, 2)
        })

    # 4. Generar el reporte agregado
    resumen_cartera = pd.DataFrame(resultados)
    
    valor_total = resumen_cartera['Valor Mercado'].sum()
    pnl_latente_total = resumen_cartera['PnL Latente'].sum()
    pnl_realizado_total = resumen_cartera['PnL Realizado'].sum()
    pnl_global_total = resumen_cartera['PnL Total'].sum()

    print("--- Resumen de la Cartera ---")
    print(resumen_cartera.to_string(index=False))
    print("-" * 35)
    print(f"Valor Total de Mercado:  {valor_total:,.2f}")
    print(f"Total PnL Latente:       {pnl_latente_total:,.2f}")
    print(f"Total PnL Realizado:     {pnl_realizado_total:,.2f}")
    print(f"PnL Global del Portfolio:{pnl_global_total:,.2f}")
    
    return resumen_cartera

# Ejecución
if __name__ == "__main__":
    # Cambia 'operaciones.csv' por el nombre real de tu archivo
    analizar_cartera('historial_operaciones.csv')