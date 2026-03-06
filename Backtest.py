import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from UniversoActivos import UniversoActivosBase
from ProveedorDatos import ProveedorDatosBase
from VariablesTransformation import FeatureEngineer
from Modelos import ModeloBase


class BacktestEngine:
    """
    Motor de backtest genérico. Orquesta el flujo completo:
    datos → features → entrenamiento → scoring → rebalanceo → métricas.

    Parámetros
    ----------
    universo         : UniversoActivosBase  — gestiona qué tickers son válidos en cada fecha
    proveedor        : ProveedorDatosBase   — datos ya descargados (df_daily, df_weekly)
    feature_engineer : FeatureEngineer      — construye features y target
    modelo           : ModeloBase           — modelo de clasificación
    start_date       : str                  — fecha de inicio del backtest
    end_date         : str                  — fecha de fin del backtest
    len_ventana      : int                  — años de historia para entrenar
    n_activos_obj    : int                  — tamaño objetivo de la cartera
    umbral_salida    : int                  — top-N para el buffer de permanencia
    """

    def __init__(
        self,
        universo: UniversoActivosBase,
        proveedor: ProveedorDatosBase,
        feature_engineer: FeatureEngineer,
        modelo: ModeloBase,
        start_date: str,
        end_date: str,
        len_ventana: int,
        n_activos_obj: int,
        umbral_salida: int,
    ):
        self.universo      = universo
        self.modelo        = modelo
        self.fe            = feature_engineer
        self.start_date    = pd.Timestamp(start_date)
        self.end_date      = pd.Timestamp(end_date)
        self.len_ventana   = len_ventana
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida

        self.df = self.fe.build(proveedor.df_weekly, proveedor.df_daily)
        self.composiciones: dict[pd.Timestamp, set] = {}

    # ------------------------------------------------------------------
    # Entrenamiento rolling
    # ------------------------------------------------------------------

    def _train(self, fecha_pivote: pd.Timestamp, tickers_validos: set) -> bool:
        fecha_inicio = fecha_pivote - pd.DateOffset(years=self.len_ventana)
        train = self.df[
            (self.df["Fecha"] >= fecha_inicio) &
            (self.df["Fecha"] <  fecha_pivote) &
            (self.df["Ticker"].isin(tickers_validos))
        ]
        if train.empty:
            return False
        self.modelo.train(train[self.fe.feature_cols], train["Target"])
        return True

    # ------------------------------------------------------------------
    # Backtest principal
    # ------------------------------------------------------------------

    def run(self, coste_operacion: float = 0.0005) -> tuple[pd.DataFrame, float]:
        fecha_inicio_bt = self.start_date + pd.DateOffset(years=self.len_ventana)
        fechas = sorted(f for f in self.df["Fecha"].unique() if f >= fecha_inicio_bt)

        # Precalcular composiciones para cada fecha
        for f in fechas:
            self.composiciones[f] = self.universo.get_universe_at_date(f)

        cartera_actual: set  = set()
        historial_neto: list = []
        fechas_plot: list    = []
        ultima_fecha_train   = None
        modelo_entrenado     = False

        for fecha_hoy in fechas[:-1]:
            tickers_validos = self.composiciones[fecha_hoy]

            # Re-entrenamiento cada 6 meses
            if ultima_fecha_train is None or (fecha_hoy - ultima_fecha_train).days >= 180:
                ok = self._train(fecha_hoy, tickers_validos)
                if ok:
                    ultima_fecha_train = fecha_hoy
                    modelo_entrenado  = True

            if not modelo_entrenado:
                continue

            datos_hoy = self.df[
                (self.df["Fecha"] == fecha_hoy) &
                (self.df["Ticker"].isin(tickers_validos))
            ].copy()

            if datos_hoy.empty:
                continue

            # Scoring
            datos_hoy["Score"] = self.modelo.predict_proba(datos_hoy[self.fe.feature_cols])
            datos_hoy = datos_hoy.sort_values("Score", ascending=False)

            # Gestión de delists
            tickers_hoy    = set(datos_hoy["Ticker"])
            tickers_delist = cartera_actual - tickers_hoy
            if tickers_delist:
                print(f"{fecha_hoy.date()}: Delisted: {tickers_delist}")
            cartera_actual = cartera_actual & tickers_hoy

            # Selección con buffer
            if not cartera_actual:
                nueva_cartera = set(datos_hoy.head(self.n_activos_obj)["Ticker"])
            else:
                top_mant      = set(datos_hoy.head(self.umbral_salida)["Ticker"])
                mantener      = cartera_actual & top_mant
                huecos        = self.n_activos_obj - len(mantener)
                candidatos    = [t for t in datos_hoy["Ticker"] if t not in mantener]
                nueva_cartera = mantener | set(candidatos[:huecos])

            # Costes de transacción
            n_ops  = len(cartera_actual - nueva_cartera) + len(nueva_cartera - cartera_actual)
            coste  = (n_ops / self.n_activos_obj) * coste_operacion

            # Retorno semanal medio de la cartera
            retorno = datos_hoy[datos_hoy["Ticker"].isin(nueva_cartera)]["Retorno_1W"].mean()

            if pd.notna(retorno):
                historial_neto.append(retorno - coste)
                fechas_plot.append(fecha_hoy)

            cartera_actual = nueva_cartera

        if not historial_neto:
            return pd.DataFrame(columns=["Fecha", "Retorno_Neto", "Curva"]), float("nan")

        resultados = pd.DataFrame({"Fecha": fechas_plot, "Retorno_Neto": historial_neto})
        resultados["Curva"] = (1 + resultados["Retorno_Neto"]).cumprod()
        rendimiento_total   = (resultados["Curva"].iloc[-1] - 1) * 100

        self._print_results(resultados)
        return resultados, rendimiento_total

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics(r: pd.Series, freq: int = 52, rf: float = 0.02) -> pd.Series:
        curva  = (1 + r).cumprod()
        cagr   = curva.iloc[-1] ** (freq / len(r)) - 1
        vol    = r.std() * np.sqrt(freq)
        sharpe = (r.mean() * freq - rf) / vol if vol > 0 else float("nan")
        dd     = (curva / curva.cummax() - 1).min()
        return pd.Series({"Total": curva.iloc[-1]-1, "CAGR": cagr, "Vol": vol,
                          "Sharpe": sharpe, "MaxDD": dd, "Hit": (r > 0).mean()})

    def _print_results(self, resultados: pd.DataFrame) -> None:
        ret_ml = resultados.set_index("Fecha")["Retorno_Neto"]
        ret_ml.index = pd.to_datetime(ret_ml.index)

        # Benchmark EW dinámico
        r_next = self.df.pivot_table(index="Fecha", columns="Ticker", values="Retorno_1W")
        r_next.index = pd.to_datetime(r_next.index)
        r_next = r_next.reindex(ret_ml.index)

        ret_bh = pd.Series(
            [
                r_next.loc[f, r_next.loc[f].index.isin(self.composiciones.get(f, set()))].dropna().mean() or 0
                for f in r_next.index
            ],
            index=r_next.index,
        )

        # Tabla de métricas
        tabla = pd.concat([self._metrics(ret_ml), self._metrics(ret_bh)], axis=1)
        tabla.columns = ["ML", "B&H EW"]
        fmt = tabla.copy()
        for c in ["Total", "CAGR", "Vol", "MaxDD", "Hit"]:
            fmt.loc[c] = fmt.loc[c].map(lambda x: f"{x:.2%}")
        fmt.loc["Sharpe"] = fmt.loc["Sharpe"].map(lambda x: f"{x:.2f}")
        print("=== Métricas ==="); display(fmt)

        # Rentabilidad anual
        anual = pd.concat(
            [(1+ret_ml).resample("Y").prod()-1, (1+ret_bh).resample("Y").prod()-1], axis=1
        )
        anual.columns = ["ML", "B&H EW"]
        anual.index   = anual.index.year
        print("=== Rentabilidad Anual ==="); display(anual.style.format("{:.2%}"))

        # Gráfico
        plt.figure(figsize=(12, 5))
        plt.plot((1+ret_ml).cumprod(), label="ML",     lw=2)
        plt.plot((1+ret_bh).cumprod(), label="B&H EW", lw=2, ls="--")
        plt.title("ML vs Buy&Hold EW"); plt.xlabel("Fecha"); plt.ylabel("Multiplicador")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()