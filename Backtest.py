import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from UniversoActivos import UniversoActivosBase
from ProveedorDatos import ProveedorDatosBase
from VariablesTransformation import FeatureEngineer
from Modelos import ModeloBase
from Estrategia import EstrategiaBase
from auxiliary_functions import build_metrics_table, calcular_costes, mark_to_market

class BacktestEngine:
    def __init__(self, universo: UniversoActivosBase, proveedor: ProveedorDatosBase,
                    feature_engineer: FeatureEngineer, estrategia: EstrategiaBase,
                    start_date: str, end_date: str, len_ventana: int,
                    nominal: float):
        self.universo    = universo
        self.proveedor   = proveedor
        self.estrategia  = estrategia
        self.fe          = feature_engineer
        self.end_date    = pd.Timestamp(end_date)
        self.len_ventana = len_ventana
        self.posicion    = {} # diccionario ticker -> cantidad de acciones, actualizado cada fecha
        self.VP          = nominal # Valor presente de la cartera

        # Esto es para asegurarnos de que el backtest empieza un viernes
        self.start_date = pd.Timestamp(start_date)
        dias_hasta_viernes = (4 - self.start_date.weekday()) % 7  # 4 = viernes
        if dias_hasta_viernes > 0:
            self.start_date += pd.DateOffset(days=dias_hasta_viernes)

        all_tickers = self.universo.get_full_ticker_list()
        self.costes = calcular_costes(all_tickers) # Costes de transacción por ticker (estáticos)
        data_start_date = self.start_date - pd.DateOffset(years=self.len_ventana + 1)
        self.df_daily = proveedor.download_prices_daily(all_tickers, data_start_date, end_date)
        df_weekly   = proveedor.download_prices_weekly(all_tickers, data_start_date, end_date)
        self.df = self.fe.build(df_weekly, self.df_daily) # Df con toda la informacion necesaria para cada fecha y ticker
        
    def _train(self, fecha_pivote: pd.Timestamp, tickers_validos: set) -> bool:
        fecha_inicio = fecha_pivote - pd.DateOffset(years=self.len_ventana)
        train_data = self.df[(self.df["Fecha"] >= fecha_inicio) & (self.df["Fecha"] < fecha_pivote)]
        train_daily = self.df_daily[(self.df_daily["Fecha"] >= fecha_inicio) & (self.df_daily["Fecha"] < fecha_pivote)]

        return self.estrategia.train(train_data, self.fe.feature_cols, tickers_validos, train_daily)

    def _ajustar_pesos(self, cartera: dict, precios_hoy) -> dict[str, float]:
        pesos = {}
        for ticker, cantidad in cartera.items():
            if ticker == "cash":
                pesos["cash"] = cantidad / self.VP
                continue

            precio = precios_hoy.get(ticker)
            if pd.isna(precio):
                # No podemos operar con este ticker hoy, lo ignoramos
                continue
            pesos[ticker] = cantidad * precio / self.VP

        return pesos
    
    def _ajustar_cartera(self, cartera: dict, datos_hoy: pd.DataFrame,
                         pesos_nuevos: dict) -> tuple[dict, dict, float]:
        '''
        Calcula la cartera ajustada, nuevos pesos y valor total de la cartera tras realizar un ajuste
        según unos pesos nuevos que se quieren obtener.
        '''
        precios_hoy = datos_hoy.set_index("Ticker")["Precio_Close"]
        pesos_antiguos = self._ajustar_pesos(cartera, precios_hoy) # Calculamos los pesos reales antes del ajuste

        for ticker in set(cartera.keys()) | set(pesos_nuevos.keys()):
            if ticker == "cash":
                continue

            precio = precios_hoy.get(ticker)
            if pd.isna(precio): #REVISAR
                # No podemos operar con este ticker hoy, lo ignoramos
                continue

            if ticker not in pesos_nuevos: # vendemos todo
                precio_venta = precio * (1 - self.costes[ticker])
                cartera["cash"] += cartera[ticker] * precio_venta
                cartera.pop(ticker, None)
                pesos_antiguos.pop(ticker, None)
            elif ticker not in pesos_antiguos: # compramos nuevo activo
                precio_compra = precio * (1 + self.costes[ticker])
                cartera[ticker] = math.floor((pesos_nuevos[ticker] * self.VP) / precio_compra)
                cartera["cash"] -= cartera[ticker] * precio_compra
            else: # ajustamos posición existente
                ajuste = pesos_nuevos[ticker] - pesos_antiguos[ticker]
                if ajuste > 0: # aumento de posición
                    precio_compra = precio * (1 + self.costes[ticker])
                    cant_compra = math.floor((ajuste * self.VP) / precio_compra)
                    cartera[ticker] += cant_compra
                    cartera["cash"] -= cant_compra * precio_compra
                elif ajuste < 0: # reducción de posición
                    precio_venta = precio * (1 - self.costes[ticker])
                    cant_venta = math.ceil((-ajuste * self.VP) / precio_venta)
                    cartera[ticker] -= cant_venta
                    cartera["cash"] += cant_venta * precio_venta

        # Calculamos el valor total de la cartera
        valor_total = cartera["cash"]
        for ticker, cantidad in cartera.items():
            if ticker == "cash":
                continue
            valor_total += cantidad * precios_hoy.get(ticker, np.nan)

        # Calculamos los pesos de cada activo tras el ajuste
        pesos_adj = self._ajustar_pesos(cartera, precios_hoy)

        return cartera, pesos_adj, valor_total
    
    def _run(self) -> tuple[pd.DataFrame, float]:
        fecha_inicio_bt = self.start_date
        fecha_fin_bt = self.end_date
        # fechas semanales
        fechas = sorted(f for f in self.df["Fecha"].unique() if (f >= fecha_inicio_bt and f <= fecha_fin_bt))

        fechas_diarias = sorted(f for f in self.df_daily["Fecha"].unique() if (f >= fecha_inicio_bt and f <= fecha_fin_bt))

        historial_neto = {}
        ultima_fecha_train = None
        modelo_entrenado = False
        cartera = {"cash": self.VP} # Empezamos 100% en cash

        for fecha_hoy in fechas_diarias:
            datos_hoy = self.df_daily[self.df_daily["Fecha"] == fecha_hoy].copy()
            self.VP = mark_to_market(cartera, datos_hoy)
            historial_neto[fecha_hoy] = self.VP

            if fecha_hoy in fechas[:-1]:
                tickers_hoy = self.universo.get_universe_at_date(fecha_hoy)

                # Re-entrenamiento cada 6 meses
                if ultima_fecha_train is None or (fecha_hoy - ultima_fecha_train).days >= 180:
                    train_flag = self._train(fecha_hoy, tickers_hoy)
                    if train_flag:
                        ultima_fecha_train = fecha_hoy
                        modelo_entrenado = True
                        #print(f"Modelo entrenado en fecha {fecha_hoy.date()}")

                if not modelo_entrenado:
                    print(f"fallo al entrenar el modelo a fecha {fecha_hoy.date()}")
                    continue

                # La estrategia decide el reajuste de pesos. Actualizamos la cartera y los pesos
                datos_features_hoy = self.df[self.df["Fecha"] == fecha_hoy].copy()
                pesos_nuevos = self.estrategia.seleccionar(datos_features_hoy, self.fe.feature_cols, cartera, self.df_daily)
                print(f"{fecha_hoy.date()} | VP={self.VP:.0f} | pesos={pesos_nuevos}")
                cartera, pesos, self.VP = self._ajustar_cartera(cartera, datos_hoy, pesos_nuevos)
                
                historial_neto[fecha_hoy] = self.VP

        return pd.DataFrame(list(historial_neto.items()), columns=["Fecha", "Valor cartera"])
    
    def print_results(self, bmks: list | None = None, bmk_equal_weight: list | None = None,
                  plot: bool = True, oracle: bool = False) -> None:
        serie_estrategia = self._run().set_index("Fecha")["Valor cartera"]
        serie_estrategia = serie_estrategia / serie_estrategia.iloc[0]
        fechas = serie_estrategia.index

        series_comp = {"Estrategia": serie_estrategia}

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(fechas, serie_estrategia, label="Estrategia", linewidth=2)

        if bmks is None:
            bmks = []

        for bmk in bmks:
            df_bmk = self.proveedor.download_prices_daily(bmk, self.start_date.strftime("%Y-%m-%d"),
                                                    self.end_date.strftime("%Y-%m-%d"))
            df_bmk["Fecha"] = pd.to_datetime(df_bmk["Fecha"])
            primera_inversion = serie_estrategia[serie_estrategia != serie_estrategia.iloc[0]].index[0]
            serie_bmk = df_bmk.set_index("Fecha")["Precio_Close"].reindex(fechas).ffill()
            serie_bmk = serie_bmk / serie_bmk.loc[primera_inversion]
            series_comp[bmk] = serie_bmk
            if plot:
                plt.plot(fechas, serie_bmk, label=bmk, linestyle="--")

        if bmk_equal_weight:
            df_ew = self.proveedor.download_prices_daily(bmk_equal_weight,
                                                        self.start_date.strftime("%Y-%m-%d"),
                                                        self.end_date.strftime("%Y-%m-%d"))
            precios = (df_ew.assign(Fecha=pd.to_datetime(df_ew["Fecha"]))
                    .pivot(index="Fecha", columns="Ticker", values="Precio_Close").sort_index())
            serie_ew = (1 + precios.pct_change().mean(axis=1, skipna=True).fillna(0)).cumprod()
            serie_ew = serie_ew.reindex(fechas).ffill()
            serie_ew = serie_ew / serie_ew.iloc[0]
            series_comp["Benchmark EW"] = serie_ew
            if plot:
                plt.plot(fechas, serie_ew, label="Benchmark EW", linestyle=":")

        if plot and oracle:
            serie_oraculo = self._serie_oraculo()
            serie_oraculo = serie_oraculo / serie_oraculo.iloc[0]
            plt.plot(serie_oraculo.index, serie_oraculo, label="Oráculo Top15", 
                    linestyle="-.", color="gold")

        if plot:
            plt.title("Evolución de la cartera vs Benchmarks")
            plt.xlabel("Fecha")
            plt.legend()
            plt.show()

        metrics_view = build_metrics_table(series_comp, periods_per_year=252, rf_annual=0.0).T

        pct_rows = {
            "Rentabilidad total", "Rentabilidad anualizada", "Volatilidad anualizada",
            "Max Drawdown", "Win rate", "Mejor periodo", "Peor periodo"
        }
        metrics_view = metrics_view.apply(
            lambda row: row.map(lambda x: f"{x:.2%}") if row.name in pct_rows
            else row.map(lambda x: f"{x:.2f}"),
            axis=1
        )

        if plot:
            display(metrics_view)

        return fechas, serie_estrategia, metrics_view
    
    def _serie_oraculo(self, n: int = 15) -> pd.Series:
        fechas = sorted(f for f in self.df["Fecha"].unique() 
                        if (f >= self.start_date and f <= self.end_date))
        
        precios = (self.df[self.df["Fecha"].isin(fechas)]
                .pivot(index="Fecha", columns="Ticker", values="Precio_Close")
                .sort_index())

        retornos = precios.pct_change()
        vp = 1
        historial = {}

        for i, fecha in enumerate(fechas[1:], 1):
            top15 = retornos.loc[fecha].nlargest(n).index.tolist()
            retorno_medio = retornos.loc[fecha, top15].mean()
            vp *= (1 + retorno_medio)
            historial[fecha] = vp

        return pd.Series(historial)
    
# Añadir a Backtest.py o crear BacktestRandom.py

class BacktestRandom:
    """Backtest rápido de una estrategia aleatoria:
    - elige n_activos al azar
    - asigna pesos aleatorios entre 2% y 20%
    - mantiene la cartera una semana
    - aplica costes según turnover
    """

    def __init__(self, universo, proveedor, start_date: str, end_date: str,
                 nominal: float, n_activos: int = 15):
        self.universo = universo
        self.proveedor = proveedor
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.nominal = nominal
        self.n_activos = n_activos
        self.peso_min = 0.02
        self.peso_max = 0.20

        dias_hasta_viernes = (4 - self.start_date.weekday()) % 7
        if dias_hasta_viernes > 0:
            self.start_date += pd.DateOffset(days=dias_hasta_viernes)

        tickers = universo.get_full_ticker_list()
        self.costes = calcular_costes(tickers)

        self.df = proveedor.download_prices_weekly(
            tickers,
            self.start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
        )
        self.df["Fecha"] = pd.to_datetime(self.df["Fecha"])

        self.precios = (
            self.df.pivot(index="Fecha", columns="Ticker", values="Precio_Close")
            .sort_index()
        )

        self.fechas = [f for f in self.precios.index if self.start_date <= f <= self.end_date]
        if len(self.fechas) < 2:
            raise ValueError("No hay suficientes fechas para ejecutar el backtest.")

    def _generar_pesos(self, n: int, rng) -> np.ndarray:
        while True:
            w = rng.dirichlet(np.ones(n))
            if (w >= self.peso_min).all() and (w <= self.peso_max).all():
                return w

    def _cartera_aleatoria(self, fecha, rng) -> dict[str, float]:
        tickers_validos = list(self.universo.get_universe_at_date(fecha))
        precios_hoy = self.precios.loc[fecha]

        tickers_validos = [t for t in tickers_validos if t in precios_hoy.index and pd.notna(precios_hoy[t])]
        if not tickers_validos:
            return {}

        n = min(self.n_activos, len(tickers_validos))
        elegidos = rng.choice(tickers_validos, size=n, replace=False)
        pesos = self._generar_pesos(n, rng)

        return {t: float(p) for t, p in zip(elegidos, pesos)}

    def _coste_cambio(self, cartera_ant, cartera_nueva) -> float:
        tickers = set(cartera_ant) | set(cartera_nueva)
        coste = 0.0
        for t in tickers:
            delta = abs(cartera_nueva.get(t, 0.0) - cartera_ant.get(t, 0.0))
            coste += delta * self.costes.get(t, 0.0005)
        return coste

    def _retorno_cartera(self, cartera, fecha, fecha_sig) -> float:
        if not cartera:
            return 0.0

        precios_0 = self.precios.loc[fecha]
        precios_1 = self.precios.loc[fecha_sig]

        retorno = 0.0
        suma_pesos = 0.0

        for t, w in cartera.items():
            p0 = precios_0.get(t, np.nan)
            p1 = precios_1.get(t, np.nan)
            if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                retorno += w * (p1 / p0 - 1)
                suma_pesos += w

        return retorno / suma_pesos if suma_pesos > 0 else 0.0

    def _run_once(self, seed: int | None = None) -> pd.Series:
        rng = np.random.default_rng(seed)
        vp = self.nominal
        historial = {self.fechas[0]: vp}
        cartera_ant = {}

        for i in range(len(self.fechas) - 1):
            fecha = self.fechas[i]
            fecha_sig = self.fechas[i + 1]

            cartera = self._cartera_aleatoria(fecha, rng)
            coste = self._coste_cambio(cartera_ant, cartera)
            retorno = self._retorno_cartera(cartera, fecha, fecha_sig)

            vp *= (1 - coste)
            vp *= (1 + retorno)

            historial[fecha_sig] = vp
            cartera_ant = cartera

        return pd.Series(historial).sort_index()

    def run_montecarlo(self, n_sims: int = 1000, benchmark: str | None = None) -> dict:
        series = [self._run_once(seed=i) for i in range(n_sims)]
        df_sims = pd.concat(series, axis=1)
        df_sims.columns = [f"sim_{i}" for i in range(n_sims)]

        resultados = {
            "media": df_sims.mean(axis=1),
            "std": df_sims.std(axis=1),
            "p10": df_sims.quantile(0.10, axis=1),
            "p50": df_sims.quantile(0.50, axis=1),
            "p90": df_sims.quantile(0.90, axis=1),
            "todas": df_sims,
        }

        if benchmark:
            df_bmk = self.proveedor.download_prices_weekly(
                [benchmark],
                self.start_date.strftime("%Y-%m-%d"),
                self.end_date.strftime("%Y-%m-%d"),
            )
            df_bmk["Fecha"] = pd.to_datetime(df_bmk["Fecha"])
            serie_bmk = (
                df_bmk.set_index("Fecha")["Precio_Close"]
                .sort_index()
                .reindex(self.fechas)
                .ffill()
            )
            serie_bmk = serie_bmk / serie_bmk.iloc[0] * self.nominal
            resultados["benchmark"] = serie_bmk

        return resultados