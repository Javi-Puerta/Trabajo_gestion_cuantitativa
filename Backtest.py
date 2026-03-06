import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from UniversoActivos import UniversoActivosBase
from ProveedorDatos import ProveedorDatosBase
from VariablesTransformation import FeatureEngineer
from Modelos import ModeloBase
from Estrategia import EstrategiaBase


class BacktestEngine:
    class BacktestEngine:
        def __init__(
            self,
            universo: UniversoActivosBase,
            proveedor: ProveedorDatosBase,
            feature_engineer: FeatureEngineer,
            estrategia: EstrategiaBase,        # ← sustituye a modelo + params de selección
            start_date: str,
            end_date: str,
            len_ventana: int,
        ):
            self.universo    = universo
            self.estrategia  = estrategia
            self.fe          = feature_engineer
            self.start_date  = pd.Timestamp(start_date)
            self.end_date    = pd.Timestamp(end_date)
            self.len_ventana = len_ventana

            self.df = self.fe.build(proveedor.df_weekly, proveedor.df_daily)
            self.composiciones: dict[pd.Timestamp, set] = {}

        def _train(self, fecha_pivote: pd.Timestamp, tickers_validos: set) -> bool:
            fecha_inicio = fecha_pivote - pd.DateOffset(years=self.len_ventana)
            train = self.df[
                (self.df["Fecha"] >= fecha_inicio) &
                (self.df["Fecha"] <  fecha_pivote)
            ]
            return self.estrategia.train(train, self.fe.feature_cols, tickers_validos)

        def run(self, coste_operacion: float = 0.001) -> tuple[pd.DataFrame, float]:
            fecha_inicio_bt = self.start_date + pd.DateOffset(years=self.len_ventana)
            fechas = sorted(f for f in self.df["Fecha"].unique() if f >= fecha_inicio_bt)

            for f in fechas:
                self.composiciones[f] = self.universo.get_universe_at_date(f)

            historial_neto: list = []
            fechas_plot: list    = []
            ultima_fecha_train   = None
            modelo_entrenado     = False
            pesos_anteriores: dict = {}

            for fecha_hoy in fechas[:-1]:
                tickers_validos = self.composiciones[fecha_hoy]

                # Re-entrenamiento cada 6 meses
                if ultima_fecha_train is None or (fecha_hoy - ultima_fecha_train).days >= 180:
                    ok = self._train(fecha_hoy, tickers_validos)
                    if ok:
                        ultima_fecha_train = fecha_hoy
                        modelo_entrenado   = True

                if not modelo_entrenado:
                    continue

                datos_hoy = self.df[self.df["Fecha"] == fecha_hoy].copy()
                if datos_hoy.empty:
                    continue

                # La estrategia decide los pesos
                pesos_nuevos = self.estrategia.seleccionar(datos_hoy, self.fe.feature_cols, tickers_validos)
                if not pesos_nuevos:
                    continue

                # Costes de transacción
                tickers_anteriores = set(pesos_anteriores.keys())
                tickers_nuevos     = set(pesos_nuevos.keys())
                n_ops  = len(tickers_anteriores - tickers_nuevos) + len(tickers_nuevos - tickers_anteriores)
                coste  = (n_ops / len(tickers_nuevos)) * coste_operacion if tickers_nuevos else 0

                # Retorno ponderado
                retornos = datos_hoy.set_index("Ticker")["Retorno_1W"]
                retorno  = sum(pesos_nuevos[t] * retornos[t] for t in tickers_nuevos if t in retornos)

                if pd.notna(retorno):
                    historial_neto.append(retorno - coste)
                    fechas_plot.append(fecha_hoy)

                pesos_anteriores = pesos_nuevos

            if not historial_neto:
                return pd.DataFrame(columns=["Fecha", "Retorno_Neto", "Curva"]), float("nan")

            resultados = pd.DataFrame({"Fecha": fechas_plot, "Retorno_Neto": historial_neto})
            resultados["Curva"] = (1 + resultados["Retorno_Neto"]).cumprod()
            rendimiento_total   = (resultados["Curva"].iloc[-1] - 1) * 100

            self._print_results(resultados)
            return resultados, rendimiento_total