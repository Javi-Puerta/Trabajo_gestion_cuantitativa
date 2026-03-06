import pandas as pd

from abc import ABC, abstractmethod

class EstrategiaBase(ABC):
    @abstractmethod
    def run_backtest(self, coste_operacion: float = 0.001) -> tuple[pd.DataFrame, float]:
        pass

class Estrategia1(EstrategiaBase):
    pass