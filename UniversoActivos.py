
import yfinance as yf
import pandas as pd

from abc import ABC, abstractmethod

class UniversoActivosBase(ABC):
    """
    Responsabilidad única: saber qué tickers son válidos en una fecha dada
    y cuál es el universo histórico completo (para descargar precios).
    """

    @abstractmethod
    def get_full_ticker_list(self) -> list[str]:
        """Unión de todos los tickers que han formado parte del universo."""
        ...

    @abstractmethod
    def get_universe_at_date(self, date: pd.Timestamp) -> set[str]:
        """Tickers válidos en esa fecha concreta."""
        ...

class UniversoActivosEstatico(UniversoActivosBase):
    """
    Universo fijo: la lista de tickers nunca cambia.
    Adecuado para carteras de fondos, o ETFs.
    """
    def __init__(self, tickers: list[str]):
        self.tickers = tickers

    def get_full_ticker_list(self) -> list[str]:
        return self.tickers

    def get_universe_at_date(self, date: pd.Timestamp) -> set[str]:
        return set(self.tickers)

class UniversoActivosDinamico(UniversoActivosBase):
    """
    Universo variable: la lista de tickers puede cambiar con el tiempo. Los cambios están en un CSV
    con columnas date, Tickr added, Tickr removed.
    Adecuado para carteras de activos de un índice como el S&P 500.
    """
    def __init__(self, tickers_actuales: list[str], start_date, end_date, csv_cambios_path):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self._tickers_actuales = set(tickers_actuales)
        self._cambios = self._load_changes(csv_cambios_path)

    def _load_changes(self, csv_cambios_path) -> pd.DataFrame:
        df = pd.read_csv(csv_cambios_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= self.start_date]

        return df.sort_values("date").dropna(subset=["date"])

    def get_full_ticker_list(self) -> list[str]:
        todos = self._tickers_actuales.copy()
        for _, row in self._cambios[self._cambios["date"] >= self.start_date].iterrows():
            if pd.notna(row["Tickr removed"]):
                todos.add(row["Tickr removed"])

        return sorted(todos)
    
    def get_universe_at_date(self, date: pd.Timestamp) -> set[str]:
        tickers = self._tickers_actuales.copy()
        for _, row in self._cambios[self._cambios["date"] > date].iterrows():
            if pd.notna(row["Tickr added"]):
                tickers.discard(row["Tickr added"])
            if pd.notna(row["Tickr removed"]):
                tickers.add(row["Tickr removed"])

        return tickers