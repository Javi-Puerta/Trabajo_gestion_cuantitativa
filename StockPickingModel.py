import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from auxiliary_functions import calculate_rsi, calculate_macd

class StockPickingModel:
    def __init__(self, start_date, end_date, criterio, len_ventana,
                 n_activos_obj, umbral_salida):
        self.start_date = start_date
        self.end_date = end_date
        self.criterio = criterio # Puede ser un entero (Top N) o 'mediana'. Se usa para definir el target
        self.len_ventana = len_ventana
        self.model = None
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self.composiciones = {}  # {fecha: set(tickers)}
        
        # Descargar cambios históricos y calcular tickers
        self._cambios_sp500, self._tickers_actuales = self._descargar_cambios_sp500()
        self.ticker_list = self._obtener_todos_tickers()
        
        self.df = self.obtain_data()
        self.df = self.obtain_variables()

    def _descargar_cambios_sp500(self):
        '''Descarga datos de Wikipedia una sola vez.'''
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text)
        
        tickers_actuales = set(tables[0]["Symbol"].astype(str).str.strip())
        
        df_chg = tables[1].copy()
        if isinstance(df_chg.columns, pd.MultiIndex):
            df_chg.columns = [" ".join(str(x) for x in c if str(x) != "nan").strip().lower() for c in df_chg.columns]
        else:
            df_chg.columns = [str(c).lower() for c in df_chg.columns]
        
        date_col = next(c for c in df_chg.columns if "date" in c)
        add_col = next(c for c in df_chg.columns if "added" in c)
        rem_col = next(c for c in df_chg.columns if "removed" in c)
        
        df_chg = df_chg.rename(columns={date_col: 'date', add_col: 'added', rem_col: 'removed'})
        df_chg['date'] = pd.to_datetime(df_chg['date'], errors="coerce")
        
        return df_chg.dropna(subset=['date']), tickers_actuales

    def _composicion_en_fecha(self, target_date):
        '''Calcula composición S&P 500 para una fecha.'''
        tickers = self._tickers_actuales.copy()
        for _, row in self._cambios_sp500[self._cambios_sp500['date'] > pd.Timestamp(target_date)].iterrows():
            added, removed = str(row['added']).strip(), str(row['removed']).strip()
            if added.lower() != 'nan': tickers.discard(added)
            if removed.lower() != 'nan': tickers.add(removed)
        return tickers

    def _obtener_todos_tickers(self):
        '''Unión de todos los tickers históricos.'''
        todos = self._tickers_actuales.copy()
        for _, row in self._cambios_sp500[self._cambios_sp500['date'] >= pd.Timestamp(self.start_date)].iterrows():
            removed = str(row['removed']).strip()
            if removed.lower() != 'nan': todos.add(removed)
        return sorted(todos)

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
                                                     'Retorno_Next_Week', 'Rank_Semanal',
                                                     'Precio_Close', 'SMA_20'])

        return self.df.dropna() # Limpiamos filas con valores vacíos
    
    def obtain_transaction_costs(self):
        pass

    def train_model(self, fecha_pivote, tickers_validos):
        '''
        Entrena el modelo usando una ventana de len_ventana años previos a la fecha_pivote.
        '''
        fecha_inicio_train = pd.to_datetime(fecha_pivote) - pd.DateOffset(years=self.len_ventana)
        train_data = self.df[(self.df['Fecha'] >= fecha_inicio_train) & 
                            (self.df['Fecha'] < fecha_pivote)].copy()
        
        # Filtrar solo tickers válidos
        train_data = train_data[train_data['Ticker'].isin(tickers_validos)]
        
        if train_data.empty:
            print(f"No hay datos suficientes para entrenar antes de {fecha_pivote}")
            return None

        X = train_data[self.variables]
        y = train_data['Target']

        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42,
                                            class_weight={0: 1, 1: 10})
        self.model.fit(X, y)
        self.last_train_date = fecha_pivote

    def run_backtest(self, coste_operacion=0.001):
        '''
        Ejecuta la simulación iterando por las fechas, reentrenando cada 6 meses,
        y aplicando la lógica de rebalanceo con Buffer.
        '''
        fecha_inicio_backtest = pd.to_datetime(self.start_date) + pd.DateOffset(years=self.len_ventana)
        
        fechas_totales = sorted(self.df['Fecha'].unique())
        fechas_backtest = [f for f in fechas_totales if f >= fecha_inicio_backtest]
        
        # Precalcular composiciones para cada fecha
        for f in fechas_backtest:
            self.composiciones[f] = self._composicion_en_fecha(f)
        
        cartera_actual = set()
        historial_neto = []
        fechas_plot = []
        
        ultima_fecha_entrenamiento = None
        
        for fecha_hoy in fechas_backtest[:-1]:
            tickers_validos = self.composiciones[fecha_hoy]
            
            # Re-entrenamiento cada 6 meses
            if ultima_fecha_entrenamiento is None or (fecha_hoy - ultima_fecha_entrenamiento).days >= 180:
                self.train_model(fecha_hoy, tickers_validos)
                ultima_fecha_entrenamiento = fecha_hoy
            
            # Filtrar datos por tickers válidos en esta fecha
            datos_hoy = self.df[self.df['Fecha'] == fecha_hoy].copy()
            datos_hoy = datos_hoy[datos_hoy['Ticker'].isin(tickers_validos)]
            
            if datos_hoy.empty or self.model is None:
                continue

            # Scoring
            X_hoy = datos_hoy[self.variables]
            datos_hoy['Score'] = self.model.predict_proba(X_hoy)[:, 1]
            datos_hoy = datos_hoy.sort_values('Score', ascending=False)
            
            # Filtrar cartera por tickers que siguen siendo válidos
            cartera_actual = cartera_actual.intersection(tickers_validos)
            
            # Selección con buffer
            if len(cartera_actual) == 0:
                nuevos_elegidos = set(datos_hoy.head(self.n_activos_obj)['Ticker'])
            else:
                top_mantenimiento = set(datos_hoy.head(self.umbral_salida)['Ticker'])
                mantener = cartera_actual.intersection(top_mantenimiento)
                huecos = self.n_activos_obj - len(mantener)
                candidatos = [t for t in datos_hoy['Ticker'] if t not in mantener]
                nuevos_elegidos = set(list(mantener) + candidatos[:huecos])
            
            # Calcular operaciones y costes
            a_vender = cartera_actual - nuevos_elegidos
            a_comprar = nuevos_elegidos - cartera_actual
            num_operaciones = len(a_vender) + len(a_comprar)
            coste_total = (num_operaciones / self.n_activos_obj) * coste_operacion
            
            # Retorno de la cartera
            retorno = datos_hoy[datos_hoy['Ticker'].isin(nuevos_elegidos)]['Retorno_Next_Week'].mean()
            
            if pd.notna(retorno):
                historial_neto.append(retorno - coste_total)
                fechas_plot.append(fecha_hoy)
            
            cartera_actual = nuevos_elegidos
        
        # Construir resultados
        if len(historial_neto) == 0:
            return pd.DataFrame(columns=['Fecha', 'Retorno_Neto', 'Curva']), np.nan
        
        resultados = pd.DataFrame({
            'Fecha': fechas_plot,
            'Retorno_Neto': historial_neto
        })
        resultados['Curva'] = (1 + resultados['Retorno_Neto']).cumprod()
        rendimiento_total = (resultados['Curva'].iloc[-1] - 1) * 100
        
        self.results_backtest(resultados)
        return resultados, rendimiento_total
    
    def results_backtest(self, resultados):
        from IPython.display import display
        
        # Retornos ML
        ret_ml = resultados.set_index("Fecha")["Retorno_Neto"].copy()
        ret_ml.index = pd.to_datetime(ret_ml.index)
        
        # Benchmark EW con composición dinámica
        r_next = self.df.pivot_table(index="Fecha", columns="Ticker", values="Retorno_Next_Week")
        r_next.index = pd.to_datetime(r_next.index)
        r_next = r_next.loc[ret_ml.index]
        
        ret_bh = []
        for fecha in r_next.index:
            tickers_validos = self.composiciones.get(fecha, set())
            retornos_fecha = r_next.loc[fecha].dropna()
            retornos_validos = retornos_fecha[retornos_fecha.index.isin(tickers_validos)]
            ret_bh.append(retornos_validos.mean() if len(retornos_validos) > 0 else 0)
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