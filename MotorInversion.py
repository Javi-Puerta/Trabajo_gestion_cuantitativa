from __future__ import annotations
import json
import pickle, warnings
from datetime import date, timedelta
from pathlib import Path
from auxiliary_functions import calcular_costes, mark_to_market

import numpy as np
import pandas as pd
import math

class MotorInversion:
    CARTERA_FILE    = "cartera_actual.json"
    HISTORIAL_FILE  = "historial_operaciones.csv"
    TRAIN_DATE_FILE = "ultimo_entrenamiento.txt"
    MODELO_FILE     = "modelo_estado.pkl"
    DIV_DATE_FILE   = "ultima_fecha_dividendos.txt"
    MESES_RETRAIN   = 6

    def __init__(self, universo, feature_engineer, estrategia, estado_path, len_ventana,
                 capital_total, proveedor_cls):

        self.universo   = universo
        self.tickers    = self.universo.get_full_ticker_list()
        self.costes     = calcular_costes(self.tickers)
        self.fe         = feature_engineer
        self.estrategia = estrategia
        self.path       = Path(estado_path)
        self.ventana    = len_ventana
        self.capital    = capital_total
        self.proveedor_cls = proveedor_cls
        self.path.mkdir(parents=True, exist_ok=True)
        self.cartera = self._cargar_cartera()
        self._ultimo_train = self._leer_fecha_train()
        self._ultima_fecha_dividendos = self._leer_fecha_dividendos()   
        self._cargar_modelo()


    def ejecutar(self, fecha: date) -> pd.DataFrame:
        '''Ejecuta el motor de inversión. Reentrena el modelo si toca (cada 6 meses), genera las señales
        (nuevos pesos) y guarda los datos de compra y posición actual de la cartera en documentos csv.'''        
        self._reentrenar_si_toca(fecha)
        señales = self._generar_señales(fecha)
        self._guardar_cartera()
        self._guardar_historial(fecha, señales)
        self._guardar_modelo()
        print(señales.to_string(index=False))

        return señales


    def _reentrenar_si_toca(self, fecha: date):
        necesita = (self._ultimo_train is None or
                    (fecha - self._ultimo_train).days >= self.MESES_RETRAIN * 30)
        if not necesita:
            return
        
        df_daily, df_weekly = self._descargar_datos(fecha, long_hist=self.ventana)
        df = self.fe.build(df_weekly, df_daily)
        fecha_corte = df["Fecha"].max() - pd.Timedelta(weeks=2)
        df_train = df[df["Fecha"] <= fecha_corte]
        tickers = list(self.universo.get_universe_at_date(fecha))
        self.estrategia.train(df_train, self.fe.feature_cols, set(tickers), df_daily)
        self._ultimo_train = fecha_corte.date()
        (self.path / self.TRAIN_DATE_FILE).write_text(self._ultimo_train.isoformat())


    def _descargar_datos(self, fecha: date, long_hist):
        '''Descarga los datos de precios desde long_hist años antes de la fecha. Necesitamos un año
        más de datos para poder construir bien las variables.'''
        start = (fecha - pd.DateOffset(years=long_hist + 1)).strftime("%Y-%m-%d")
        end = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
        tickers  = list(set(self.tickers + [self.fe.ticker_indice]))
        prov = self.proveedor_cls()

        return (prov.download_prices_daily(tickers, start, end),
                prov.download_prices_weekly(tickers, start, end))
    
    
    def _leer_fecha_dividendos(self) -> date | None:
        p = self.path / self.DIV_DATE_FILE
        return date.fromisoformat(p.read_text().strip()) if p.exists() else None


    def _guardar_fecha_dividendos(self, fecha: date):
        (self.path / self.DIV_DATE_FILE).write_text(fecha.isoformat())


    def _actualizar_dividendos(self, fecha: pd.Timestamp, df_daily: pd.DataFrame) -> float:
        if "Dividendos" not in df_daily.columns:
            raise ValueError("df_daily no tiene columna 'Dividendos'. Revisa ProveedorDatos.py.")

        fecha = pd.Timestamp(fecha).normalize()

        if self._ultima_fecha_dividendos is None:
            self._ultima_fecha_dividendos = fecha.date()
            self._guardar_fecha_dividendos(fecha.date())
            return 0.0

        fecha_ini = pd.Timestamp(self._ultima_fecha_dividendos)
        tickers = [t for t in self.cartera if t != "cash"]

        if not tickers:
            self._ultima_fecha_dividendos = fecha.date()
            self._guardar_fecha_dividendos(fecha.date())
            return 0.0

        mask = (
            df_daily["Ticker"].isin(tickers)
            & (df_daily["Fecha"] > fecha_ini)
            & (df_daily["Fecha"] <= fecha)
        )

        divs = df_daily.loc[mask].groupby("Ticker")["Dividendos"].sum()

        cobrado = sum(
            float(self.cartera.get(t, 0.0)) * float(divs.get(t, 0.0))
            for t in tickers
        )

        self.cartera["cash"] = float(self.cartera.get("cash", 0.0)) + cobrado
        self._ultima_fecha_dividendos = fecha.date()
        self._guardar_fecha_dividendos(fecha.date())

        return float(cobrado)


    def _generar_señales(self, fecha: date) -> pd.DataFrame:
        fecha = pd.Timestamp(fecha)

        df_daily, df_weekly = self._descargar_datos(fecha, long_hist=self.ventana)

        div_cobrados = self._actualizar_dividendos(fecha, df_daily)
        if div_cobrados != 0:
            print(f"[Dividendos] {fecha.date()} | cobrados: {div_cobrados:.2f}€")

        df = self.fe.build(df_weekly, df_daily)
        tickers_validos = self.universo.get_universe_at_date(fecha)
        df_hoy = df[(df["Fecha"] == fecha) & (df["Ticker"].isin(tickers_validos))]

        datos_valoracion = df_daily[df_daily["Fecha"] == fecha].copy()
        precios = datos_valoracion.set_index("Ticker")["Precio_Close"].to_dict()

        pesos_obj = self.estrategia.seleccionar(
            df_hoy,
            self.fe.feature_cols,
            self.cartera,
            df_daily
        )

        print(f"[Señales] {fecha.date()} | Pesos objetivo: {pesos_obj}")

        actuales = set(self.cartera.keys()) - {"cash"}
        objetivo = set(pesos_obj.keys())
        filas = []

        vp = mark_to_market(self.cartera, datos_valoracion)
        print(f"Valor de la cartera antes de ejecutar señales: {vp:.2f}€")

        # Ventas
        for ticker in actuales - objetivo:
            coste = self.costes.get(ticker, 0.0005)
            precio = precios.get(ticker, np.nan)
            uds = self.cartera.get(ticker, 0.0)
            precio_ejec = precio * (1 - coste)
            self.cartera["cash"] += uds * precio_ejec
            self.cartera.pop(ticker)
            filas.append({"Ticker": ticker, "Accion": "VENTA", "Cantidad": int(uds), "Precio": precio,
                          "CT": precio * coste, "Precio_Ejecutado": precio_ejec})

        # Compras
        for ticker in objetivo - actuales:
            coste = self.costes.get(ticker, 0.0005)
            peso = pesos_obj[ticker]
            precio = precios.get(ticker, np.nan)
            precio_ejec = precio * (1 + coste)
            uds = math.floor((vp * peso) / precio_ejec)
            self.cartera["cash"] -= uds * precio_ejec
            self.cartera[ticker] = uds
            filas.append({"Ticker": ticker, "Accion": "COMPRA", "Cantidad": int(uds), "Precio": precio,
                         "CT": precio * coste, "Precio_Ejecutado": precio_ejec})

        # Mantener y ajustar pesos
        for ticker in objetivo & actuales:
            coste = self.costes.get(ticker, 0.0005)
            precio = precios.get(ticker, np.nan)
            uds_antiguas = self.cartera[ticker]
            peso_antiguo = uds_antiguas * precio / vp
            peso = pesos_obj[ticker] - peso_antiguo
            if peso > 0:
                accion = "COMPRA"
                precio_ejec = precio * (1 + coste)
                uds = math.floor(abs(vp * peso) / precio_ejec)
                uds_nueva = uds_antiguas + uds
                self.cartera["cash"] -= uds * precio_ejec
            elif peso < 0:
                accion = "VENTA"
                precio_ejec = precio * (1 - coste)
                uds = math.floor(abs(vp * peso) / precio_ejec)
                uds = min(uds, uds_antiguas)
                uds_nueva = uds_antiguas - uds
                self.cartera["cash"] += uds * precio_ejec
            else:
                accion = "MANTENER"
                precio_ejec = precio
                uds_nueva = uds_antiguas
                uds = 0
                coste = 0.0
            if uds_nueva == 0:
                self.cartera.pop(ticker, None)
            else:
                self.cartera[ticker] = uds_nueva
            filas.append({"Ticker": ticker, "Accion": accion, "Cantidad": int(abs(uds)), "Precio": precio,
                         "CT": precio * coste, "Precio_Ejecutado": precio_ejec})

        for t, q in self.cartera.items(): #Comprobación
            if t != "cash" and q < 0:
                raise ValueError(f"Posición negativa detectada: {t} = {q}")
            
        return pd.DataFrame(filas).sort_values("Accion").reset_index(drop=True)


    def _cargar_cartera(self) -> pd.DataFrame:
        '''Carga la cartera actual desde un archivo json.'''
        p = self.path / self.CARTERA_FILE
        if not p.exists():
            return {"cash": self.capital}
        with open(p, "r") as f:
            return json.load(f)


    def _guardar_cartera(self):
        '''Guarda la cartera actual en un archivo json.'''
        p = self.path / self.CARTERA_FILE
        with open(p, "w") as f:
            json.dump(self.cartera, f)


    def _guardar_historial(self, fecha: date, señales: pd.DataFrame):
        if señales.empty:
            return
        p  = self.path / self.HISTORIAL_FILE
        df = señales.copy()
        df.insert(0, "fecha", fecha)
        df.to_csv(p, mode="a", header=not p.exists(), index=False)


    def _leer_fecha_train(self) -> date | None:
        '''Lee la fecha del último entrenamiento del modelo.'''
        p = self.path / self.TRAIN_DATE_FILE
        
        return date.fromisoformat(p.read_text().strip()) if p.exists() else None


    def _cargar_modelo(self):
        p = self.path / self.MODELO_FILE
        if not p.exists():
            return
        with open(p, "rb") as f:
            modelo = pickle.load(f)
        if hasattr(self.estrategia, "modelo"):
            self.estrategia.modelo = modelo


    def _guardar_modelo(self):
        modelo = getattr(self.estrategia, "modelo", None)
        if modelo is None:
            return
        with open(self.path / self.MODELO_FILE, "wb") as f:
            pickle.dump(modelo, f)
            
