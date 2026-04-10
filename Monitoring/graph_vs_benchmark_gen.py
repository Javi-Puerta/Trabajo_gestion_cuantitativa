import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def plot_cartera_vs_benchmark(csv_path, benchmark_ticker='^STOXX50E'):
    # 1. Cargar y preparar datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: No se encontró el archivo CSV.")
        return

    df.columns = df.columns.str.strip().str.lower()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')

    # 2. Rango temporal y descarga de datos (Cartera + Benchmark)
    fecha_inicio = df['fecha'].min()
    fecha_fin = pd.Timestamp.today()
    
    tickers_cartera = df['ticker'].unique().tolist()
    tickers_descarga = tickers_cartera + [benchmark_ticker]
    
    print(f"Descargando históricos desde {fecha_inicio.date()}...")
    # Usamos auto_adjust=False para evitar problemas de compatibilidad en algunas versiones de yf
    datos_mercado = yf.download(tickers_descarga, start=fecha_inicio, end=fecha_fin)['Close']
    
    # Forward fill para cubrir fines de semana y festivos (arrastramos el último precio de cierre)
    datos_mercado = datos_mercado.ffill()

    # 3. Calcular la Posición Diaria y los Flujos de Caja (Cashflows)
    fechas_mercado = datos_mercado.index
    
    # Matriz de posiciones (cantidad de acciones mantenidas al final del día)
    posiciones = pd.DataFrame(0.0, index=fechas_mercado, columns=tickers_cartera)
    # Vector de flujos de caja (Cashflows) netos diarios
    flujos_caja = pd.Series(0.0, index=fechas_mercado)

    for _, row in df.iterrows():
        fecha_op = row['fecha']
        ticker = row['ticker']
        accion = str(row['accion']).strip().lower()
        cantidad = float(row['cantidad'])
        precio_ejecutado = float(row['precio_ejecutado'])
        
        # Encontramos el índice válido más cercano en los días de mercado
        idx_fecha = fechas_mercado[fechas_mercado >= fecha_op]
        if len(idx_fecha) == 0: continue
        fecha_aplicacion = idx_fecha[0]

        if accion == 'compra':
            posiciones.loc[fecha_aplicacion:, ticker] += cantidad
            # Una compra es una inyección de capital a la cartera de acciones
            flujos_caja.loc[fecha_aplicacion] += (cantidad * precio_ejecutado)
            
        elif accion == 'venta':
            posiciones.loc[fecha_aplicacion:, ticker] -= cantidad
            # Una venta es una extracción de capital (realizamos caja)
            flujos_caja.loc[fecha_aplicacion] -= (cantidad * precio_ejecutado)

    # 4. Calcular el Market Value (MV) diario de los activos
    # Asegurarnos de que datos_mercado[tickers_cartera] sea un DataFrame bidimensional
    precios_activos = datos_mercado[tickers_cartera]
    if isinstance(precios_activos, pd.Series):
        precios_activos = precios_activos.to_frame()
        
    market_value = (posiciones * precios_activos).sum(axis=1)

    # 5. Calcular Time-Weighted Return (TWR) diario
    # Fórmula: R_t = (MV_t - CF_t - MV_{t-1}) / MV_{t-1}
    twr_diario = pd.Series(0.0, index=fechas_mercado)
    mv_previo = market_value.shift(1).fillna(0)

    for t in fechas_mercado:
        mv_t = market_value.loc[t]
        mv_t1 = mv_previo.loc[t]
        cf_t = flujos_caja.loc[t]
        
        if mv_t1 > 0:
            # Rendimiento asumiendo que el flujo de caja ocurre al final del día
            retorno = (mv_t - cf_t - mv_t1) / mv_t1
            twr_diario.loc[t] = retorno
        elif mv_t > 0 and mv_t1 == 0:
            # Primer día de inversión
            twr_diario.loc[t] = 0.0

    # Construir el Índice Base 100 de la cartera
    indice_cartera = 100 * (1 + twr_diario).cumprod()

    # 6. Preparar el Benchmark (Euro Stoxx 50)
    # Lo normalizamos también a Base 100 desde el primer día que tenemos cartera
    primer_dia_inversion = market_value[market_value > 0].index[0]
    benchmark_series = datos_mercado.loc[primer_dia_inversion:, benchmark_ticker]
    indice_benchmark = 100 * (benchmark_series / benchmark_series.iloc[0])
    
    # Ajustar el índice de la cartera para que empiece el mismo día
    indice_cartera = indice_cartera.loc[primer_dia_inversion:]
    # Recalibrar a 100 en esa fecha inicial por seguridad
    indice_cartera = 100 * (indice_cartera / indice_cartera.iloc[0])

    # 7. Graficar los resultados
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(indice_cartera.index, indice_cartera, label='Mi Cartera (TWR Base 100)', color='#1E88E5', linewidth=2)
    ax.plot(indice_benchmark.index, indice_benchmark, label='Euro Stoxx 50 (Base 100)', color='#FFC107', linewidth=2)

    ax.set_title('Evolución de Cartera vs Benchmark (Euro Stoxx 50)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Índice Normalizado', fontsize=12)
    
    # Formateo visual
    ax.fill_between(indice_cartera.index, indice_cartera, 100, where=(indice_cartera >= 100), color='#1E88E5', alpha=0.1)
    ax.fill_between(indice_cartera.index, indice_cartera, 100, where=(indice_cartera < 100), color='red', alpha=0.1)
    ax.axhline(100, color='black', linestyle='--', linewidth=1)
    
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    
    # Guardar o mostrar gráfico
    plt.savefig('portfolio_vs_benchmark.png', dpi=300)
    print("\nGráfico guardado exitosamente como 'portfolio_vs_benchmark.png'")
    plt.show()

if __name__ == "__main__":
    plot_cartera_vs_benchmark('historial_operaciones.csv')