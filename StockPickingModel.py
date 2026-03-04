import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from auxiliary_functions import calculate_rsi, calculate_macd

class StockPickingModel:
    def __init__(self, ticker_list, start_date, end_date, criterio, len_ventana,
                 n_activos_obj, umbral_salida):
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.criterio = criterio # Puede ser un entero (Top N) o 'mediana'. Se usa para definir el target
        self.len_ventana = len_ventana
        self.model = None
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self.df = self.obtain_data()
        self.df = self.obtain_variables()
        self.transaction_costs = self.obtain_transaction_costs()
    def obtain_data(self):
        '''
        Descargamos los datos y nos quedamos con los precios de cierre semanales. Transformamos los
        datos para poder usarlos como input de ML.
        '''
        data = yf.download(self.ticker_list, start=self.start_date, end=self.end_date,
                           interval="1d", auto_adjust=True)
        precios_close = data['Close']
        precios_semanales = precios_close.resample('W-WED').last() # Datos semanales

        # Transformación de datos
        df_final = precios_semanales.stack().reset_index()
        df_final.columns = ['Fecha', 'Ticker', 'Precio_Close']

        return df_final.sort_values(by=['Ticker', 'Fecha'])
    
    def obtain_variables(self):
        '''
        Calculamos las variables independientes y el target que usaremos en el modelo de ML. Podemos
        elegir un criterio de ranking (Top N) o usar la mediana semanal para definir el target.
        '''
        # Retorno semanal, momentum 4 semanas y Volatilidad
        self.df['Retorno_1W'] = self.df.groupby('Ticker')['Precio_Close'].pct_change(1)
        self.df['Momentum_4W'] = self.df.groupby('Ticker')['Precio_Close'].pct_change(4)
        self.df['Volatilidad_4W'] = self.df.groupby('Ticker')['Retorno_1W'].transform(lambda x: x.rolling(4).std())

        # Relación con Media Móvil de 20 semanas
        self.df['SMA_20'] = self.df.groupby('Ticker')['Precio_Close'].transform(lambda x: x.rolling(20).mean())
        self.df['Distancia_SMA'] = self.df['Precio_Close'] / self.df['SMA_20']

        # RSI y MACD
        self.df['RSI'] = self.df.groupby('Ticker')['Precio_Close'].transform(lambda x: calculate_rsi(x))
        self.df['MACD'] = self.df.groupby('Ticker')['Precio_Close'].transform(lambda x: calculate_macd(x))

        # Retorno de la semana siguiente y definición del target (variable dependiente)
        self.df['Retorno_Next_Week'] = self.df.groupby('Ticker')['Retorno_1W'].shift(-1)

        if isinstance(self.criterio, str):
            self.df['Mediana_Semanal'] = self.df.groupby('Fecha')['Retorno_Next_Week'].transform('median')
            self.df['Target'] = (self.df['Retorno_Next_Week'] > self.df['Mediana_Semanal']).astype(int)
        else:
            self.df['Rank_Semanal'] = self.df.groupby('Fecha')['Retorno_Next_Week'].rank(method='first', ascending=False)
            self.df['Target'] = (self.df['Rank_Semanal'] <= self.criterio).astype(int)

        self.variables = self.df.columns.difference(['Fecha', 'Ticker', 'Target', 'Mediana_Semanal',
                                                     'Retorno_Next_Week', 'Rank_Semanal'])

        return self.df.dropna() # Limpiamos filas con valores vacíos
    
    def obtain_transaction_costs(self):
        pass

    def train_model(self, fecha_pivote):
        '''
        Entrena el modelo usando una ventana de len_ventana años previos a la fecha_pivote.
        '''
        fecha_inicio_train = pd.to_datetime(fecha_pivote) - pd.DateOffset(years=self.len_ventana)
        train_data = self.df[(self.df['Fecha'] >= fecha_inicio_train) & 
                             (self.df['Fecha'] < fecha_pivote)].copy()
        
        if train_data.empty:
            print(f"No hay datos suficientes para entrenar antes de {fecha_pivote}")
            return None

        X = train_data[self.variables]
        y = train_data['Target']

        # Configurar y entrenar el modelo
        self.model = model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42,
                                                    class_weight={0: 1, 1: 10})
        self.model.fit(X, y)
        
        # Guardamos la fecha del último entrenamiento para saber cuándo toca el siguiente
        self.last_train_date = fecha_pivote

    def run_backtest(self, coste_operacion=0.001):
        '''
        Ejecuta la simulación iterando por las fechas, reentrenando cada 6 meses,
        y aplicando la lógica de rebalanceo con Buffer.
        '''
        fecha_inicio_backtest = pd.to_datetime(self.start_date) + pd.DateOffset(years=self.len_ventana)
        
        # Filtramos solo las fechas válidas para el backtest
        fechas_totales = sorted(self.df['Fecha'].unique())
        fechas_backtest = [f for f in fechas_totales if f >= fecha_inicio_backtest]
        
        cartera_actual = set()
        self.historial_neto = []
        self.fechas_plot = []
        
        ultima_fecha_entrenamiento = None
        
        for fecha_hoy in fechas_backtest[:-1]:
            # Re-entrenamiento cada 6 meses
            if ultima_fecha_entrenamiento is None or (fecha_hoy - ultima_fecha_entrenamiento).days >= 180:
                self.train_model(fecha_hoy)
                ultima_fecha_entrenamiento = fecha_hoy
            
            datos_hoy = self.df[self.df['Fecha'] == fecha_hoy].copy()
            if datos_hoy.empty or self.model is None:
                continue # Siguiente fecha si no hay datos o el modelo no está entrenado

            X_hoy = datos_hoy[self.variables]
            datos_hoy['Score'] = self.model.predict_proba(X_hoy)[:, 1]
            datos_hoy = datos_hoy.sort_values('Score', ascending=False)
            
            if len(cartera_actual) == 0: # Primera compra
                nuevos_elegidos = set(datos_hoy.head(self.n_activos_obj)['Ticker'].tolist())
            else:
                # A. Candidatos para mantener (Top M)
                top_mantenimiento = set(datos_hoy.head(self.umbral_salida)['Ticker'].tolist())
                quedan_en_cartera = cartera_actual.intersection(top_mantenimiento)
                
                # B. Rellenar huecos libres con los mejores del ranking
                huecos_libres = self.n_activos_obj - len(quedan_en_cartera)
                candidatos_nuevos = [t for t in datos_hoy['Ticker'].tolist() if t not in quedan_en_cartera]
                nuevos_elegidos = set(list(quedan_en_cartera) + candidatos_nuevos[:huecos_libres])
                
            # Operaciones y costes
            a_vender = cartera_actual - nuevos_elegidos
            a_comprar = nuevos_elegidos - cartera_actual
            num_operaciones = len(a_vender) + len(a_comprar)

            coste_total = (num_operaciones / self.n_activos_obj) * coste_operacion
            retorno_promedio_semanal = datos_hoy[datos_hoy['Ticker'].isin(nuevos_elegidos)]['Retorno_Next_Week'].mean()

            if pd.isna(retorno_promedio_semanal):
                continue

            retorno_final = retorno_promedio_semanal - coste_total
            self.historial_neto.append(retorno_final)
            self.fechas_plot.append(fecha_hoy)

            cartera_actual = nuevos_elegidos

        # Devolver resultados para poder testear fácilmente
        if len(self.historial_neto) == 0:
            return pd.DataFrame(columns=["Fecha", "Retorno_Neto", "Curva"]), np.nan

        resultados = pd.DataFrame({
            "Fecha": self.fechas_plot,
            "Retorno_Neto": self.historial_neto
        })
        resultados["Curva"] = (1 + resultados["Retorno_Neto"]).cumprod()
        rendimiento_total = (resultados["Curva"].iloc[-1] - 1) * 100

        self.results_backtest(resultados)
        return resultados, rendimiento_total
    
    def results_backtest(self, resultados):
        import matplotlib.pyplot as plt
        from IPython.display import display
        
        # Retornos ML
        ret_ml = resultados.set_index("Fecha")["Retorno_Neto"].copy()
        ret_ml.index = pd.to_datetime(ret_ml.index)
        
        # Benchmark Buy&Hold EW
        r_next = self.df.pivot_table(index="Fecha", columns="Ticker", values="Retorno_Next_Week")
        r_next.index = pd.to_datetime(r_next.index)
        r_next = r_next.loc[ret_ml.index]
        
        activos_ini = r_next.iloc[0].dropna().index.tolist()
        w = pd.Series(1.0 / len(activos_ini), index=activos_ini)
        
        ret_bh = []
        for fecha in r_next.index:
            r = r_next.loc[fecha].reindex(w.index).fillna(0.0)
            ret_bh.append((w * r).sum())
            w = w * (1 + r)
            w = w / w.sum() if w.sum() > 0 else w
        ret_bh = pd.Series(ret_bh, index=r_next.index)
        
        # Métricas
        def metrics(r, freq=52, rf=0.02):
            curva = (1 + r).cumprod()
            cagr = curva.iloc[-1] ** (freq / len(r)) - 1
            vol = r.std() * np.sqrt(freq)
            sharpe = (r.mean() * freq - rf) / vol if vol > 0 else np.nan
            dd = (curva / curva.cummax() - 1).min()
            return pd.Series({"Total": curva.iloc[-1]-1, "CAGR": cagr, "Vol": vol, 
                            "Sharpe": sharpe, "MaxDD": dd, "Hit": (r>0).mean()})
        
        tabla = pd.concat([metrics(ret_ml), metrics(ret_bh)], axis=1)
        tabla.columns = ["ML", "B&H EW"]
        
        tabla_fmt = tabla.copy()
        for c in ["Total", "CAGR", "Vol", "MaxDD", "Hit"]:
            tabla_fmt.loc[c] = tabla_fmt.loc[c].map(lambda x: f"{x:.2%}")
        tabla_fmt.loc["Sharpe"] = tabla_fmt.loc["Sharpe"].map(lambda x: f"{float(x.strip('%'))/100:.2f}" if '%' in str(x) else f"{x:.2f}")
        
        print("=== Métricas ===")
        display(tabla_fmt)
        
        # Rentabilidad anual
        anual = pd.concat([(1+ret_ml).resample("Y").prod()-1, (1+ret_bh).resample("Y").prod()-1], axis=1)
        anual.columns = ["ML", "B&H EW"]
        anual.index = anual.index.year
        print("=== Rentabilidad Anual ===")
        display(anual.style.format("{:.2%}"))
        
        # Gráfico
        plt.figure(figsize=(12,5))
        plt.plot((1+ret_ml).cumprod(), label="ML", lw=2)
        plt.plot((1+ret_bh).cumprod(), label="B&H EW", lw=2, ls="--")
        plt.title("ML vs Buy&Hold EW")
        plt.xlabel("Fecha"); plt.ylabel("Multiplicador")
        plt.legend(); plt.grid(alpha=0.3); plt.show()