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
                    nominal: float, frec_reentreno: int = 180):
        self.universo    = universo
        self.proveedor   = proveedor
        self.estrategia  = estrategia
        self.fe          = feature_engineer
        self.end_date    = pd.Timestamp(end_date)
        self.len_ventana = len_ventana
        self.frec_reentreno = frec_reentreno
        self.posicion    = {} # diccionario ticker -> cantidad de acciones, actualizado cada fecha
        self.VP          = nominal # Valor presente de la cartera

        # Esto es para asegurarnos de que el backtest empieza un viernes
        self.start_date = pd.Timestamp(start_date)
        dias_hasta_viernes = (4 - self.start_date.weekday()) % 7  # 4 = viernes
        if dias_hasta_viernes > 0:
            self.start_date += pd.DateOffset(days=dias_hasta_viernes)

        self.nominal = nominal

        self.tickers_invertibles = self.universo.get_full_ticker_list()
        self.all_tickers = sorted(set(self.tickers_invertibles) | {self.fe.ticker_indice})
        self.costes = calcular_costes(self.tickers_invertibles)

        data_start_date = self.start_date - pd.DateOffset(years=self.len_ventana + 1)

        self.df_daily = proveedor.download_prices_daily(
            self.all_tickers,
            data_start_date,
            end_date
        )

        self.df_weekly = proveedor.download_prices_weekly(
            self.all_tickers,
            data_start_date,
            end_date
        )
        
    
    def _datos_asof(self, fecha: pd.Timestamp):
        df_daily = self.df_daily[self.df_daily["Fecha"] <= fecha].copy()
        df_weekly = self.df_weekly[self.df_weekly["Fecha"] <= fecha].copy()
        df = self.fe.build(df_weekly, df_daily)
        return df, df_daily

        
    def _train(self, fecha_pivote: pd.Timestamp, tickers_validos: set, df_asof: pd.DataFrame, df_daily_asof: pd.DataFrame) -> bool:

        fecha_corte = fecha_pivote - pd.Timedelta(weeks=2)
        fecha_inicio = fecha_corte - pd.DateOffset(years=self.len_ventana)

        train_data = df_asof[
            (df_asof["Fecha"] >= fecha_inicio)
            & (df_asof["Fecha"] <= fecha_corte)
            & (df_asof["Ticker"].isin(tickers_validos))
        ].copy()

        train_daily = df_daily_asof[
            (df_daily_asof["Fecha"] >= fecha_inicio)
            & (df_daily_asof["Fecha"] <= fecha_corte)
        ].copy()

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


    def _cobrar_dividendos(self, cartera: dict, datos_hoy: pd.DataFrame) -> float:
        if "Dividendos" not in datos_hoy.columns:
            return 0.0

        divs = datos_hoy.set_index("Ticker")["Dividendos"]
        cobrado = 0.0

        for ticker, cantidad in cartera.items():
            if ticker == "cash":
                continue
            cobrado += float(cantidad) * float(divs.get(ticker, 0.0))

        cartera["cash"] = float(cartera.get("cash", 0.0)) + cobrado
        return cobrado


    def _ajustar_cartera(self, cartera: dict, datos_hoy: pd.DataFrame, pesos_nuevos: dict) -> tuple[dict, dict, float]:

        precios = datos_hoy.set_index("Ticker")["Precio_Close"].to_dict()
        vp = self.VP

        actuales = set(cartera.keys()) - {"cash"}
        objetivo = set(pesos_nuevos.keys())

        # Ventas completas
        for ticker in actuales - objetivo:
            precio = precios.get(ticker, np.nan)
            if pd.isna(precio):
                continue

            coste = self.costes.get(ticker, 0.0005)
            uds = cartera.get(ticker, 0.0)
            precio_ejec = precio * (1 - coste)

            cartera["cash"] += uds * precio_ejec
            cartera.pop(ticker, None)

        # Compras nuevas
        for ticker in objetivo - actuales:
            precio = precios.get(ticker, np.nan)
            if pd.isna(precio):
                continue

            coste = self.costes.get(ticker, 0.0005)
            peso = pesos_nuevos[ticker]
            precio_ejec = precio * (1 + coste)

            uds = math.floor((vp * peso) / precio_ejec)

            cartera["cash"] -= uds * precio_ejec
            cartera[ticker] = uds

        # Ajustes de posiciones existentes
        for ticker in objetivo & actuales:
            precio = precios.get(ticker, np.nan)
            if pd.isna(precio):
                continue

            coste = self.costes.get(ticker, 0.0005)
            uds_antiguas = cartera[ticker]

            peso_antiguo = uds_antiguas * precio / vp
            delta_peso = pesos_nuevos[ticker] - peso_antiguo

            if delta_peso > 0:
                precio_ejec = precio * (1 + coste)
                uds = math.floor((vp * delta_peso) / precio_ejec)

                cartera[ticker] = uds_antiguas + uds
                cartera["cash"] -= uds * precio_ejec

            elif delta_peso < 0:
                precio_ejec = precio * (1 - coste)
                uds = math.floor((-vp * delta_peso) / precio_ejec)
                uds = min(uds, uds_antiguas)

                cartera[ticker] = uds_antiguas - uds
                cartera["cash"] += uds * precio_ejec

                if cartera[ticker] == 0:
                    cartera.pop(ticker, None)

        for t, q in cartera.items():
            if t != "cash" and q < 0:
                raise ValueError(f"Posición negativa detectada en backtest: {t} = {q}")

        valor_total = mark_to_market(cartera, datos_hoy)
        precios_hoy = datos_hoy.set_index("Ticker")["Precio_Close"]
        pesos_adj = self._ajustar_pesos(cartera, precios_hoy)

        return cartera, pesos_adj, valor_total


    def _run(self) -> pd.DataFrame:
        fecha_inicio_bt = self.start_date
        fecha_fin_bt = self.end_date

        fechas_diarias = sorted(
            f for f in self.df_daily["Fecha"].unique()
            if fecha_inicio_bt <= f <= fecha_fin_bt
        )

        fechas_rebalanceo = sorted(
            f for f in self.df_weekly["Fecha"].unique()
            if fecha_inicio_bt <= f <= fecha_fin_bt
        )

        historial_neto = {}
        ultima_fecha_train = None
        modelo_entrenado = False

        cartera = {"cash": self.nominal}
        self.VP = self.nominal

        for fecha_hoy in fechas_diarias:
            datos_hoy = self.df_daily[self.df_daily["Fecha"] == fecha_hoy].copy()

            # 1) Dividendos cobrados por las posiciones existentes
            self._cobrar_dividendos(cartera, datos_hoy)

            # 2) Valoración antes de decidir
            self.VP = mark_to_market(cartera, datos_hoy)
            historial_neto[fecha_hoy] = self.VP

            # 3) Solo rebalanceamos en viernes/semanales, excepto última fecha
            if fecha_hoy not in fechas_rebalanceo[:-1]:
                continue

            df_asof, df_daily_asof = self._datos_asof(fecha_hoy)
            tickers_hoy = self.universo.get_universe_at_date(fecha_hoy)

            # 4) Reentrenamiento semestral
            if ultima_fecha_train is None or (fecha_hoy.date() - ultima_fecha_train).days >= self.frec_reentreno:
                train_flag = self._train(
                    fecha_hoy,
                    tickers_hoy,
                    df_asof,
                    df_daily_asof
                )

                if train_flag:
                    ultima_fecha_train = fecha_hoy.date()
                    modelo_entrenado = True

            if not modelo_entrenado:
                print(f"fallo al entrenar el modelo a fecha {fecha_hoy.date()}")
                continue

            # 5) Señales con información disponible en fecha_hoy
            datos_features_hoy = df_asof[
                (df_asof["Fecha"] == fecha_hoy)
                & (df_asof["Ticker"].isin(tickers_hoy))
            ].copy()

            pesos_nuevos = self.estrategia.seleccionar(
                datos_features_hoy,
                self.fe.feature_cols,
                cartera,
                df_daily_asof
            )

            print(f"{fecha_hoy.date()} | VP={self.VP:.0f} | pesos={pesos_nuevos}")

            # 6) Ejecución de cartera
            cartera, pesos, self.VP = self._ajustar_cartera(
                cartera,
                datos_hoy,
                pesos_nuevos
            )

            historial_neto[fecha_hoy] = self.VP

        return pd.DataFrame(list(historial_neto.items()), columns=["Fecha", "Valor cartera"])


    def _generar_pesos_random_sobre_activos(self, activos: list[str], rng,
                                            peso_min: float = 0.02,
                                            peso_max: float = 0.15) -> dict[str, float]:
        n = len(activos)

        if n == 0:
            return {}

        if n * peso_max < 1 or n * peso_min > 1:
            w = np.ones(n) / n
            return {t: float(p) for t, p in zip(activos, w)}

        while True:
            w = rng.dirichlet(np.ones(n))
            if (w >= peso_min).all() and (w <= peso_max).all():
                return {t: float(p) for t, p in zip(activos, w)}


    def run_con_monos_pesos(self, n_monos: int = 1000,
                            peso_min: float = 0.02,
                            peso_max: float = 0.15,
                            seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:

        fecha_inicio_bt = self.start_date
        fecha_fin_bt = self.end_date

        fechas_diarias = sorted(
            f for f in self.df_daily["Fecha"].unique()
            if fecha_inicio_bt <= f <= fecha_fin_bt
        )

        fechas_rebalanceo = sorted(
            f for f in self.df_weekly["Fecha"].unique()
            if fecha_inicio_bt <= f <= fecha_fin_bt
        )
        fechas_rebalanceo_set = set(fechas_rebalanceo[:-1])

        ultima_fecha_train = None
        modelo_entrenado = False

        cartera_real = {"cash": self.nominal}
        carteras_monos = [{"cash": self.nominal} for _ in range(n_monos)]

        hist_real = {}
        hist_monos = {i: {} for i in range(n_monos)}

        rng = np.random.default_rng(seed)

        for fecha_hoy in fechas_diarias:
            datos_hoy = self.df_daily[self.df_daily["Fecha"] == fecha_hoy].copy()

            self._cobrar_dividendos(cartera_real, datos_hoy)
            for cartera in carteras_monos:
                self._cobrar_dividendos(cartera, datos_hoy)

            self.VP = mark_to_market(cartera_real, datos_hoy)
            hist_real[fecha_hoy] = self.VP

            for i, cartera in enumerate(carteras_monos):
                hist_monos[i][fecha_hoy] = mark_to_market(cartera, datos_hoy)

            if fecha_hoy not in fechas_rebalanceo_set:
                continue

            df_asof, df_daily_asof = self._datos_asof(fecha_hoy)
            tickers_hoy = self.universo.get_universe_at_date(fecha_hoy)

            if ultima_fecha_train is None or (fecha_hoy.date() - ultima_fecha_train).days >= self.frec_reentreno:
                train_flag = self._train(
                    fecha_hoy,
                    tickers_hoy,
                    df_asof,
                    df_daily_asof
                )

                if train_flag:
                    ultima_fecha_train = fecha_hoy.date()
                    modelo_entrenado = True

            if not modelo_entrenado:
                print(f"fallo al entrenar el modelo a fecha {fecha_hoy.date()}")
                continue

            datos_features_hoy = df_asof[
                (df_asof["Fecha"] == fecha_hoy)
                & (df_asof["Ticker"].isin(tickers_hoy))
            ].copy()

            pesos_real = self.estrategia.seleccionar(
                datos_features_hoy,
                self.fe.feature_cols,
                cartera_real,
                df_daily_asof
            )

            activos_modelo = list(pesos_real.keys())

            self.VP = mark_to_market(cartera_real, datos_hoy)
            cartera_real, _, self.VP = self._ajustar_cartera(
                cartera_real,
                datos_hoy,
                pesos_real
            )
            hist_real[fecha_hoy] = self.VP

            for i, cartera in enumerate(carteras_monos):
                pesos_mono = self._generar_pesos_random_sobre_activos(
                    activos_modelo,
                    rng,
                    peso_min=peso_min,
                    peso_max=peso_max
                )

                self.VP = mark_to_market(cartera, datos_hoy)
                cartera, _, vp_mono = self._ajustar_cartera(
                    cartera,
                    datos_hoy,
                    pesos_mono
                )

                carteras_monos[i] = cartera
                hist_monos[i][fecha_hoy] = vp_mono

            print(f"{fecha_hoy.date()} | VP real={hist_real[fecha_hoy]:,.0f} | activos={activos_modelo}")

        serie_real = pd.Series(hist_real).sort_index()
        df_monos = pd.DataFrame({
            f"ml_random_{i}": pd.Series(hist_monos[i]).sort_index()
            for i in range(n_monos)
        })

        return serie_real, df_monos


    def print_results(self, bmks: list | None = None, bmk_equal_weight: list | None = None, plot: bool = True) -> None:
        serie_estrategia = self._run().set_index("Fecha")["Valor cartera"]
        serie_estrategia = serie_estrategia / self.nominal
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
            df_ew = self.proveedor.download_prices_daily(
                bmk_equal_weight,
                self.start_date.strftime("%Y-%m-%d"),
                self.end_date.strftime("%Y-%m-%d")
            )

            df_ew["Fecha"] = pd.to_datetime(df_ew["Fecha"])

            precios = (
                df_ew
                .pivot(index="Fecha", columns="Ticker", values="Precio_Close")
                .sort_index()
                .reindex(fechas)
                .ffill()
            )

            if "Dividendos" in df_ew.columns:
                dividendos = (
                    df_ew
                    .pivot(index="Fecha", columns="Ticker", values="Dividendos")
                    .reindex(precios.index)
                    .fillna(0.0)
                )
            else:
                dividendos = precios * 0.0

            ret_ew = ((precios + dividendos) / precios.shift(1) - 1).mean(axis=1, skipna=True)
            serie_ew = (1 + ret_ew.fillna(0.0)).cumprod()
            serie_ew = serie_ew / serie_ew.iloc[0]

            series_comp["Benchmark EW"] = serie_ew

            if plot:
                plt.plot(fechas, serie_ew, label="Benchmark EW", linestyle=":")

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


class BacktestRandom:
    """Backtest random consistente con BacktestEngine: acciones reales, cash, costes y dividendos."""

    _ajustar_pesos = BacktestEngine._ajustar_pesos
    _cobrar_dividendos = BacktestEngine._cobrar_dividendos
    _ajustar_cartera = BacktestEngine._ajustar_cartera

    def __init__(self, universo, proveedor, start_date: str, end_date: str,
             nominal: float, n_activos: int = 15, n_simulaciones_mc: int = 1000):
        self.universo = universo
        self.proveedor = proveedor
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.nominal = nominal
        self.n_activos = n_activos
        self.peso_min = 0.02
        self.peso_max = 0.15
        self.VP = nominal
        self.n_simulaciones_mc = n_simulaciones_mc

        dias_hasta_viernes = (4 - self.start_date.weekday()) % 7
        if dias_hasta_viernes > 0:
            self.start_date += pd.DateOffset(days=dias_hasta_viernes)

        tickers = sorted(universo.get_full_ticker_list())
        self.tickers = tickers
        self.ticker_to_i = {t: i for i, t in enumerate(tickers)}
        self.costes = calcular_costes(tickers)

        data_start_date = self.start_date - pd.DateOffset(years=1)

        self.df_daily = proveedor.download_prices_daily(
            tickers,
            data_start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
        )

        self.df_weekly = proveedor.download_prices_weekly(
            tickers,
            self.start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
        )

        self.df_daily["Fecha"] = pd.to_datetime(self.df_daily["Fecha"])
        self.df_weekly["Fecha"] = pd.to_datetime(self.df_weekly["Fecha"])

        if "Dividendos" not in self.df_daily.columns:
            self.df_daily["Dividendos"] = 0.0

        px = (
            self.df_daily
            .pivot(index="Fecha", columns="Ticker", values="Precio_Close")
            .sort_index()
            .reindex(columns=tickers)
        )

        dv = (
            self.df_daily
            .pivot(index="Fecha", columns="Ticker", values="Dividendos")
            .sort_index()
            .reindex(px.index)
            .reindex(columns=tickers)
            .fillna(0.0)
        )

        self.fechas = px.index[(px.index >= data_start_date) & (px.index <= self.end_date)]
        self.px = px.reindex(self.fechas).to_numpy(float)
        self.dv = dv.reindex(self.fechas).to_numpy(float)

        prev_px = np.roll(self.px, 1, axis=0)
        self.ret = np.nan_to_num(
            (self.px + self.dv) / prev_px - 1.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.ret[0, :] = 0.0

        self.datos_por_fecha = {
            f: g.copy() for f, g in self.df_daily.groupby("Fecha")
        }

        self.fechas_diarias = [
            f for f in self.fechas
            if self.start_date <= f <= self.end_date
        ]

        self.fechas_rebalanceo = sorted(
            f for f in self.df_weekly["Fecha"].unique()
            if self.start_date <= f <= self.end_date
        )
        self.fechas_rebalanceo_set = set(self.fechas_rebalanceo[:-1])

        if len(self.fechas_diarias) < 2:
            raise ValueError("No hay suficientes fechas para ejecutar el backtest.")


    def _generar_pesos(self, n: int, rng) -> np.ndarray:
        if n * self.peso_max < 1 or n * self.peso_min > 1:
            return np.ones(n) / n

        while True:
            w = rng.dirichlet(np.ones(n))
            if (w >= self.peso_min).all() and (w <= self.peso_max).all():
                return w


    def _run_once(self, seed: int | None = None) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(seed)

        cartera_random = {"cash": self.nominal}
        cartera_mc = {"cash": self.nominal}

        hist_random = {}
        hist_mc = {}

        for fecha in self.fechas_diarias:
            datos_hoy = self.datos_por_fecha[fecha]

            self._cobrar_dividendos(cartera_random, datos_hoy)
            self._cobrar_dividendos(cartera_mc, datos_hoy)

            vp_random = mark_to_market(cartera_random, datos_hoy)
            vp_mc = mark_to_market(cartera_mc, datos_hoy)

            hist_random[fecha] = vp_random
            hist_mc[fecha] = vp_mc

            if fecha not in self.fechas_rebalanceo_set:
                continue

            fecha_i = self.fechas.get_loc(fecha)
            pesos_random, pesos_mc = self._carteras_aleatorias_doble(fecha_i, rng)

            self.VP = vp_random
            cartera_random, _, vp_random = self._ajustar_cartera(cartera_random, datos_hoy, pesos_random)

            self.VP = vp_mc
            cartera_mc, _, vp_mc = self._ajustar_cartera(cartera_mc, datos_hoy, pesos_mc)

            hist_random[fecha] = vp_random
            hist_mc[fecha] = vp_mc

        return pd.Series(hist_random).sort_index(), pd.Series(hist_mc).sort_index()


    def run_montecarlo(self, n_sims: int = 1000, benchmark: str | None = None) -> dict:
        series = [self._run_once(seed=i) for i in range(n_sims)]

        df_random = pd.concat([s[0] for s in series], axis=1)
        df_mc = pd.concat([s[1] for s in series], axis=1)

        df_random.columns = [f"sim_{i}" for i in range(n_sims)]
        df_mc.columns = [f"sim_{i}" for i in range(n_sims)]

        def resumen(df):
            return {
                "media": df.mean(axis=1),
                "std": df.std(axis=1),
                "p10": df.quantile(0.10, axis=1),
                "p50": df.quantile(0.50, axis=1),
                "p90": df.quantile(0.90, axis=1),
                "todas": df,
            }

        resultados = {
            "pesos_random": resumen(df_random),
            "pesos_mc": resumen(df_mc),
        }

        if benchmark:
            df_bmk = self.proveedor.download_prices_daily(
                [benchmark],
                self.start_date.strftime("%Y-%m-%d"),
                self.end_date.strftime("%Y-%m-%d"),
            )
            df_bmk["Fecha"] = pd.to_datetime(df_bmk["Fecha"])

            px = (
                df_bmk.pivot(index="Fecha", columns="Ticker", values="Precio_Close")
                .sort_index()
                .reindex(df_random.index)
                .ffill()
                .bfill()
            )
            div = (
                df_bmk.pivot(index="Fecha", columns="Ticker", values="Dividendos")
                .reindex(df_random.index)
                .fillna(0.0)
            )

            ret = ((px + div) / px.shift(1) - 1).iloc[:, 0].fillna(0.0)
            resultados["benchmark"] = self.nominal * (1 + ret).cumprod()

        return resultados


    def _carteras_aleatorias_doble(self, fecha_i, rng):
        fecha = self.fechas[fecha_i]
        precios = self.px[fecha_i]

        tickers = sorted(self.universo.get_universe_at_date(fecha))
        idx = np.array([
            self.ticker_to_i[t] for t in tickers
            if t in self.ticker_to_i and np.isfinite(precios[self.ticker_to_i[t]])
        ])

        if len(idx) == 0:
            return {}, {}

        n = min(self.n_activos, len(idx))
        elegidos = rng.choice(idx, size=n, replace=False)

        pesos_random = self._generar_pesos(n, rng)
        idx_mc, pesos_mc = self._pesos_montecarlo(elegidos, fecha_i, rng)

        return (
            {self.tickers[i]: float(w) for i, w in zip(elegidos, pesos_random)},
            {self.tickers[i]: float(w) for i, w in zip(idx_mc, pesos_mc)}
        )


    def _pesos_montecarlo(self, idx, fecha_i, rng):
        ini = max(0, fecha_i - 251)
        ret = self.ret[ini:fecha_i + 1, :][:, idx]

        n = ret.shape[1]
        obs_validas = np.isfinite(ret).all(axis=1).sum()

        if n < 2 or obs_validas < 2:
            return idx, np.ones(n) / n

        media = ret.mean(axis=0)
        cov = np.cov(ret, rowvar=False)

        if not np.isfinite(cov).all():
            return idx, np.ones(n) / n

        pesos = []

        while sum(len(x) for x in pesos) < self.n_simulaciones_mc:
            raw = rng.dirichlet(np.ones(n), size=self.n_simulaciones_mc)
            w = self.peso_min + (1 - n * self.peso_min) * raw
            ok = (w <= self.peso_max).all(axis=1)
            pesos.append(w[ok])

        pesos = np.vstack(pesos)[:self.n_simulaciones_mc]

        ret_c = pesos @ media * 252
        vol_c = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos)) * np.sqrt(252)
        sharpe = np.divide(ret_c, vol_c, out=np.full_like(ret_c, -np.inf), where=vol_c > 0)

        return idx, pesos[np.argmax(sharpe)]

