import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import yfinance as yf
from Modelos import ModeloBase
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize


class EstrategiaBase(ABC):
    """
    Interfaz que cualquier estrategia debe implementar.
    Responsabilidad: decidir qué tickers comprar y con qué peso.
    """

    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set, df_daily) -> bool:
        '''Entrena el modelo y guarda los retornos históricos de los tickers válidos para la
        optimización posterior por Montecarlo.'''
        train_data = df[df["Ticker"].isin(tickers_validos)].copy()
        if train_data.empty:
            return False
        self.modelo.train(train_data, feature_cols)
        return True

    @abstractmethod
    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    cartera: dict[str, float]) -> dict[str, float]:
        """
        Devuelve un diccionario {ticker: peso} con la cartera para esa semana.
        Los pesos deben sumar 1.
        """
        ...


class EstrategiaMLEquiponderada(EstrategiaBase):
    """
    Usa un modelo ML para puntuar los activos y asigna pesos iguales
    entre los n_activos_obj mejor puntuados, con buffer de permanencia
    para reducir rotación.

    Parámetros
    ----------
    modelo        : ModeloBase — modelo de clasificación
    n_activos_obj : int        — número de activos en cartera
    umbral_salida : int        — top-N para el buffer de permanencia
    """

    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int):
        self.modelo        = modelo
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self._cartera_actual: set = set()

    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    cartera: dict[str, float], df_daily=None) -> dict[str, float]:
        datos = df_hoy.copy()

        proba = self.modelo.predict_proba(datos[feature_cols])
        datos["Score"] = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
        datos = datos.sort_values("Score", ascending=False)

        cartera_actual = set(cartera.keys()) - {"cash"}
        top = set(datos.head(self.umbral_salida)["Ticker"])
        mantener = cartera_actual & top
        huecos = self.n_activos_obj - len(mantener)
        candidatos_nuevos = [t for t in datos["Ticker"] if t not in mantener]
        candidatos = list(mantener) + candidatos_nuevos[:huecos]

        peso = 1 / len(candidatos)
        return {t: peso for t in candidatos}


class EstrategiaMLMonteCarlo(EstrategiaBase):
    '''Usa un modelo de ML para preseleccionar candidatos y luego optimiza los pesos utilizando simulaciones
    de Montecarlo, eligiendo la cartera con mayor sharpe.'''

    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int, peso_max = 0.2,
                 n_simulaciones: int = 5000, peso_min: float = 0.02, dias_retorno: int = 252):
        self.modelo         = modelo
        self.n_activos_obj   = n_activos_obj
        self.umbral_salida  = umbral_salida
        self.n_simulaciones = n_simulaciones
        self.peso_min       = peso_min
        self.peso_max       = peso_max
        self.dias_retorno = dias_retorno
        self._retornos_hist: pd.DataFrame = pd.DataFrame()        


    def seleccionar(self, df_hoy, feature_cols, cartera, df_daily=None):
        datos = df_hoy.copy()

        proba = self.modelo.predict_proba(datos[feature_cols])
        datos["Score"] = proba.values if hasattr(proba, "values") else proba
        datos = datos.sort_values("Score", ascending=False)

        cartera_actual = set(cartera.keys()) - {"cash"}
        top = set(datos.head(self.umbral_salida)["Ticker"])
        mantener = cartera_actual & top
        huecos = self.n_activos_obj - len(mantener)
        candidatos_nuevos = [t for t in datos["Ticker"] if t not in mantener]
        candidatos = list(mantener) + candidatos_nuevos[:huecos]

        # Retornos diarios actualizados a hoy
        fecha_hoy = pd.Timestamp(datos["Fecha"].iloc[0])
        hist = df_daily[df_daily["Ticker"].isin(candidatos)].copy()

        px = (
            hist.pivot(index="Fecha", columns="Ticker", values="Precio_Close")
            .sort_index()
            .loc[:fecha_hoy]
        )

        div = (
            hist.pivot(index="Fecha", columns="Ticker", values="Dividendos")
            .reindex(px.index)
            .fillna(0.0)
        )

        ret = ((px + div) / px.shift(1) - 1).tail(self.dias_retorno)
        ret = ret.dropna(axis=1, how="all").fillna(0.0)

        if ret.shape[1] < 2:
            peso = 1 / len(candidatos)
            return {t: peso for t in candidatos}

        tickers_mc = ret.columns.tolist()
        n     = len(tickers_mc)
        media = ret.mean().values
        cov   = ret.cov().values

        rng = np.random.default_rng(seed=42)
        pesos = self._generar_pesos_mc(n, rng)

        ret_c = pesos @ media * 252
        vol_c = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos)) * np.sqrt(252)

        sharpe = np.divide(
            ret_c,
            vol_c,
            out=np.full_like(ret_c, -np.inf),
            where=vol_c > 0,
        )

        best_pesos = pesos[np.argmax(sharpe)]

        return {t: float(best_pesos[i]) for i, t in enumerate(tickers_mc)}


    def _generar_pesos_mc(self, n: int, rng) -> np.ndarray:
        if n * self.peso_max < 1 or n * self.peso_min > 1:
            return np.tile(np.ones(n) / n, (self.n_simulaciones, 1))

        aceptados = []

        while sum(len(x) for x in aceptados) < self.n_simulaciones:
            raw = rng.dirichlet(np.ones(n), size=self.n_simulaciones)
            pesos = self.peso_min + (1 - n * self.peso_min) * raw
            ok = (pesos <= self.peso_max).all(axis=1)
            aceptados.append(pesos[ok])

        return np.vstack(aceptados)[:self.n_simulaciones]


class EstrategiaMLEquiponderadaMacro(EstrategiaMLEquiponderada):
    '''Extiende la estrategia de ML equiponderada con una capa adicional de gestión macro: si el 
    índice de referencia está por debajo de su media móvil de 200 días, reduce la exposición a RV 
    y asigna el resto a un activo de cobertura (hedge).'''
    def __init__(self, modelo, n_activos_obj, umbral_salida,
                 ticker_indice="^STOXX50E", umbral_vol=0.20,
                 exposicion_rv=0.5, ticker_hedge="4GLD.DE"):  # ← añadir hedge
        super().__init__(modelo, n_activos_obj, umbral_salida)
        self.ticker_indice = ticker_indice
        self.umbral_vol    = umbral_vol
        self.exposicion_rv = exposicion_rv
        self.ticker_hedge  = ticker_hedge

    def _señal_riesgo(self, fecha_hoy: pd.Timestamp, df_daily: pd.DataFrame) -> bool:
        precios = (
            df_daily[df_daily["Ticker"] == self.ticker_indice]
            .set_index("Fecha")["Precio_Close"]
            .sort_index()
            .loc[:fecha_hoy]
        )
        if len(precios) < 200:
            return False
        return float(precios.iloc[-1]) < float(precios.rolling(200).mean().iloc[-1])

    def seleccionar(self, df_hoy, feature_cols, cartera, df_daily=None):
        pesos = super().seleccionar(df_hoy, feature_cols, cartera, df_daily)

        if df_daily is not None:
            fecha_hoy = pd.Timestamp(df_hoy["Fecha"].iloc[0])
            if self._señal_riesgo(fecha_hoy, df_daily):
                peso_hedge = 1 - self.exposicion_rv
                pesos = {t: p * self.exposicion_rv for t, p in pesos.items()}
                pesos[self.ticker_hedge] = peso_hedge

        return pesos

