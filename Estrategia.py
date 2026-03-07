import pandas as pd
from abc import ABC, abstractmethod
from Modelos import ModeloBase


class EstrategiaBase(ABC):
    """
    Interfaz que cualquier estrategia debe implementar.
    Responsabilidad: decidir qué tickers comprar y con qué peso.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        """
        Entrena la estrategia con los datos del periodo.
        Devuelve True si el entrenamiento fue exitoso, False si no hay datos suficientes.
        """
        ...

    @abstractmethod
    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    tickers_validos: set) -> dict[str, float]:
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

    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        train_data = df[df["Ticker"].isin(tickers_validos)]
        if train_data.empty:
            return False
        self.modelo.train(train_data[feature_cols], train_data["Target"])
        return True

    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    cartera: dict[str, float]) -> dict[str, float]:
        datos = df_hoy.copy()
        if datos.empty:
            return {}

        proba = self.modelo.predict_proba(datos[feature_cols])
        datos["Score"] = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
        datos = datos.sort_values("Score", ascending=False)

        # Posiciones actuales (ignorando cash)
        cartera_actual = set(cartera.keys()) - {"cash"}

        tickers_hoy = set(datos["Ticker"])
        mantener_base = cartera_actual & tickers_hoy  # elimina delisted automáticamente

        if not mantener_base:
            nueva_cartera = set(datos.head(self.n_activos_obj)["Ticker"])
        else:
            top_mant = set(datos.head(self.umbral_salida)["Ticker"])
            mantener = mantener_base & top_mant
            huecos = self.n_activos_obj - len(mantener)
            candidatos = [t for t in datos["Ticker"] if t not in mantener]
            nueva_cartera = mantener | set(candidatos[:huecos])

        peso = 1 / len(nueva_cartera)
        return {t: peso for t in nueva_cartera}