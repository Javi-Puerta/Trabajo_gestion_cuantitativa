import yfinance as yf
import pandas as pd

from abc import ABC, abstractmethod

class ProveedorDatosBase(ABC):
    """
    Clase para descargar precios históricos
    """

    @abstractmethod
    def download_prices_daily(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Devuelve DataFrame con columnas ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD'] con frecuencia
        diaria, ordenado por Ticker y Fecha.
        """
        pass

    @abstractmethod
    def download_prices_weekly(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Devuelve DataFrame con columnas ['Fecha', 'Ticker', 'Precio_Close', 'Volumen_USD'] ordenado por
        Ticker y Fecha. Los precios son ajustados y semanales (último precio de cada semana).
        """
        pass

class YFinanceProvider(ProveedorDatosBase):
    def download_prices_daily(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(tickers, start=start_date, end=end_date,
                           interval="1d", auto_adjust=True)

        precios = data["Close"].stack().reset_index()
        precios.columns = ["Fecha", "Ticker", "Precio_Close"]

        volumenes = data["Volume"].stack().reset_index()
        volumenes.columns = ["Fecha", "Ticker", "Volumen"]

        df = precios.merge(volumenes, on=["Fecha", "Ticker"])
        df["Volumen_USD"] = df["Precio_Close"] * df["Volumen"]

        return df.drop(columns="Volumen").sort_values(["Ticker", "Fecha"]).reset_index(drop=True)

    def download_prices_weekly(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        df_daily = self.download_prices_daily(tickers, start_date, end_date)
        df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])

        weekly = df_daily.set_index("Fecha").groupby("Ticker").resample("W-WED")
        weekly = weekly.agg(Precio_Close=("Precio_Close", "last"), Volumen_USD=("Volumen_USD", "sum")).reset_index()

        return weekly.sort_values(["Ticker", "Fecha"]).reset_index(drop=True)