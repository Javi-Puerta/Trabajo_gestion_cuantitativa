from __future__ import annotations
import pickle, warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import math

warnings.filterwarnings("ignore")


class MotorInversion:
    """
    Uso:
        motor = MotorInversion(universo, fe, estrategia, capital_total=100_000)
        señales = motor.ejecutar()   # cada miércoles
    """

    CARTERA_FILE    = "cartera_actual.csv"
    HISTORIAL_FILE  = "historial_operaciones.csv"
    TRAIN_DATE_FILE = "ultimo_entrenamiento.txt"
    MODELO_FILE     = "modelo_estado.pkl"
    MESES_RETRAIN   = 6

    def __init__(self, universo, feature_engineer, estrategia,
                 estado_path="./estado_cartera", len_ventana=4,
                 coste_operacion=0.001, capital_total=100_000.0, proveedor_cls=None):

        self.universo   = universo
        self.fe         = feature_engineer
        self.estrategia = estrategia
        self.path       = Path(estado_path)
        self.ventana    = len_ventana
        self.coste      = coste_operacion
        self.capital    = capital_total
        self.path.mkdir(parents=True, exist_ok=True)

        if proveedor_cls is None:
            from ProveedorDatos import YFinanceProvider
            self.proveedor_cls = YFinanceProvider
        else:
            self.proveedor_cls = proveedor_cls

        self._cartera      = self._cargar_cartera()
        self._ultimo_train = self._leer_fecha_train()
        self._cargar_modelo()

    # ── API pública ────────────────────────────────────────────────────────────

    def ejecutar(self, fecha: date | None = None) -> pd.DataFrame:
        fecha = fecha or date.today()
        df_daily, df_weekly = self._descargar_datos(fecha)
        print(f"Compras hechas a fecha", df_daily["Fecha"].max())
        df = self.fe.build(df_weekly, df_daily)
        if df.empty:
            return pd.DataFrame()
        self._reentrenar_si_toca(fecha, df)
        señales = self._generar_señales(fecha, df)
        self._guardar_cartera()
        self._guardar_historial(fecha, señales)
        self._guardar_modelo()
        print(señales.to_string(index=False))
        return señales

    def get_cartera(self) -> pd.DataFrame:
        return self._cartera.copy()

    def get_historial(self) -> pd.DataFrame:
        p = self.path / self.HISTORIAL_FILE
        return pd.read_csv(p, parse_dates=["fecha"]) if p.exists() else pd.DataFrame()

    def marcar_ejecutada(self, ticker: str, precio_real: float, unidades: float):
        mask = self._cartera["ticker"] == ticker
        self._cartera.loc[mask, "precio_entrada_real"] = precio_real
        self._cartera.loc[mask, "unidades"]            = unidades
        self._guardar_cartera()

    # ── Datos ─────────────────────────────────────────────────────────────────

    def _descargar_datos(self, fecha: date):
        start = (fecha - pd.DateOffset(years=self.ventana)).strftime("%Y-%m-%d")
        end = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
        tickers  = list(set(self.universo.get_full_ticker_list() + [self.fe.ticker_indice]))
        prov = self.proveedor_cls()
        return (prov.download_prices_daily(tickers, start, end), 
                prov.download_prices_weekly(tickers, start, end))

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def _reentrenar_si_toca(self, fecha: date, df: pd.DataFrame):
        necesita = (self._ultimo_train is None or
                    (fecha - self._ultimo_train).days >= self.MESES_RETRAIN * 30)
        if not necesita:
            return
        fecha_corte = df["Fecha"].max() - pd.Timedelta(weeks=2)
        df_train = df[df["Fecha"] <= fecha_corte]
        tickers = list(self.universo.get_universe_at_date(fecha))
        self.estrategia.train(df_train, self.fe.feature_cols, tickers)
        self._ultimo_train = fecha_corte.date()
        (self.path / self.TRAIN_DATE_FILE).write_text(self._ultimo_train.isoformat())

    # ── Señales ───────────────────────────────────────────────────────────────

    def _generar_señales(self, fecha: date, df: pd.DataFrame) -> pd.DataFrame:
        fecha_max       = df["Fecha"].max()
        tickers_validos = self.universo.get_universe_at_date(fecha)
        df_hoy          = df[(df["Fecha"] == fecha_max) & (df["Ticker"].isin(tickers_validos))]
        pesos_obj       = self.estrategia.seleccionar(df_hoy, self.fe.feature_cols, self._cartera_como_dict())
        precios         = df_hoy.set_index("Ticker")["Precio_Close"].to_dict()

        actuales = set(self._cartera["ticker"].tolist()) - {"cash"}
        objetivo = set(pesos_obj.keys())
        filas    = []
        vp       = self._mark_to_market(precios)

        # Ventas
        for ticker in actuales - objetivo:
            precio      = precios.get(ticker, np.nan)
            uds         = self._unidades(ticker)
            precio_ejec = round(precio * (1 - self.coste), 4)
            self._actualizar_cash(uds * precio_ejec)
            self._cartera = self._cartera[self._cartera["ticker"] != ticker]
            filas.append({"Ticker": ticker, "Accion": "VENTA", "Cantidad": int(uds),
                        "Precio": precio, "CT": round(precio * self.coste, 4),
                        "Precio_Ejecutado": precio_ejec})

        # Compras
        for ticker in objetivo - actuales:
            peso        = pesos_obj[ticker]
            precio      = precios.get(ticker, np.nan)
            precio_ejec = round(precio * (1 + self.coste), 4)
            uds         = math.floor((vp * peso) / precio_ejec) if not np.isnan(precio) else 0
            self._actualizar_cash(-uds * precio_ejec)
            nueva = {"ticker": ticker, "fecha_entrada": fecha, "peso_objetivo": peso,
                    "precio_entrada_ref": precio, "precio_entrada_real": np.nan, "unidades": uds}
            self._cartera = pd.concat([self._cartera, pd.DataFrame([nueva])], ignore_index=True)
            filas.append({"Ticker": ticker, "Accion": "COMPRA", "Cantidad": uds,
                        "Precio": precio, "CT": round(precio * self.coste, 4),
                        "Precio_Ejecutado": precio_ejec})

        # Mantener
        for ticker in objetivo & actuales:
            precio = precios.get(ticker, np.nan)
            filas.append({"Ticker": ticker, "Accion": "MANTENER", "Cantidad": int(self._unidades(ticker)),
                        "Precio": precio, "CT": 0.0, "Precio_Ejecutado": precio})

        return pd.DataFrame(filas).sort_values("Accion").reset_index(drop=True)

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _cargar_cartera(self) -> pd.DataFrame:
        p = self.path / self.CARTERA_FILE
        if p.exists():
            return pd.read_csv(p, parse_dates=["fecha_entrada"])
        return pd.DataFrame([{
            "ticker": "cash", "fecha_entrada": date.today(), "peso_objetivo": 1.0,
            "precio_entrada_ref": self.capital, "precio_entrada_real": self.capital, "unidades": self.capital
        }])

    def _guardar_cartera(self):
        self._cartera.to_csv(self.path / self.CARTERA_FILE, index=False)

    def _guardar_historial(self, fecha: date, señales: pd.DataFrame):
        if señales.empty:
            return
        p  = self.path / self.HISTORIAL_FILE
        df = señales.copy()
        df.insert(0, "fecha", fecha)
        df.to_csv(p, mode="a", header=not p.exists(), index=False)

    def _leer_fecha_train(self) -> date | None:
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

    def _unidades(self, ticker: str) -> float:
        if self._cartera.empty:
            return 0.0
        fila = self._cartera[self._cartera["ticker"] == ticker]
        return float(fila["unidades"].iloc[0]) if not fila.empty else 0.0
    
    def _cartera_como_dict(self) -> dict:
        if self._cartera.empty:
            return {"cash": self.capital}
        cash_row = self._cartera[self._cartera["ticker"] == "cash"]
        cash = float(cash_row["unidades"].iloc[0]) if not cash_row.empty else 0.0
        tickers = self._cartera[self._cartera["ticker"] != "cash"]
        cartera = dict(zip(tickers["ticker"], tickers["unidades"]))
        cartera["cash"] = cash
        return cartera
    
    def _mark_to_market(self, precios: dict) -> float:
        cartera = self._cartera_como_dict()
        vp = cartera.get("cash", 0.0)
        for ticker, uds in cartera.items():
            if ticker == "cash":
                continue
            vp += uds * precios.get(ticker, 0.0)
        return vp

    def _actualizar_cash(self, delta: float):
        mask = self._cartera["ticker"] == "cash"
        self._cartera.loc[mask, "unidades"] += delta