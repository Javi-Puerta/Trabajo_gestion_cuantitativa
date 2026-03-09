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


class EstrategiaMLBase(EstrategiaBase):
    """
    Clase base para estrategias basadas en ML que puntúan activos.
    """
    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int, umbral_target: float = -float('inf')):
        self.modelo        = modelo
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self.umbral_target = umbral_target
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
        
        # Filtrar solo activos con Score >= umbral_target
        if self.umbral_target > -float('inf'):
            datos = datos[datos["Score"] >= self.umbral_target]
            if datos.empty:
                return {}

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

        peso = 1 / len(nueva_cartera) if nueva_cartera else 0
        return {t: peso for t in nueva_cartera}


class EstrategiaMLEquiponderada(EstrategiaMLBase):
    """
    Usa un modelo ML para puntuar los activos y asigna pesos iguales
    entre los n_activos_obj mejor puntuados, con buffer de permanencia
    para reducir rotación.
    """
    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int):
        super().__init__(modelo, n_activos_obj, umbral_salida, umbral_target=-float('inf'))


class EstrategiaMarkI(EstrategiaMLBase):
    """
    Usa un modelo ML para puntuar los activos. 
    Elige el top-N mejor puntuados y que a la vez tienen una probabilidad de superar al mercado (columna "Target") mayor a 0.75. 
    Asigna pesos iguales entre los seleccionados, con buffer de permanencia para reducir rotación.
    """
    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int, umbral_target: float = 0.75):
        super().__init__(modelo, n_activos_obj, umbral_salida, umbral_target=umbral_target)