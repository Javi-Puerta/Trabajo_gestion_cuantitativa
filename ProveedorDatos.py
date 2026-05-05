import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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
                interval="1d", auto_adjust=False, progress=False)

        precios = data["Adj Close"].stack().reset_index()
        precios.columns = ["Fecha", "Ticker", "Precio_Close"]

        volumenes = data["Volume"].stack().reset_index()
        volumenes.columns = ["Fecha", "Ticker", "Volumen"]

        df = precios.merge(volumenes, on=["Fecha", "Ticker"])
        df["Volumen_USD"] = df["Precio_Close"] * df["Volumen"]
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        
        # Crear índice completo de fechas
        fecha_inicio = pd.to_datetime(start_date)
        fecha_fin = pd.to_datetime(end_date)
        todas_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
        todos_tickers = df["Ticker"].unique()
        
        idx_completo = pd.MultiIndex.from_product([todas_fechas, todos_tickers], 
                                                names=["Fecha", "Ticker"])
        
        df = df.set_index(["Fecha", "Ticker"]).reindex(idx_completo)
        
        # Forward fill y BACKWARD fill para asegurar que todos tengan precio desde el día 1
        df["Precio_Close"] = df.groupby("Ticker")["Precio_Close"].ffill()
        df["Volumen_USD"] = df.groupby("Ticker")["Volumen_USD"].ffill()
        
        df = df.reset_index()
        df = df.dropna(subset=["Precio_Close"])

        return df[["Fecha", "Ticker", "Precio_Close", "Volumen_USD"]].sort_values(
            ["Ticker", "Fecha"]
        ).reset_index(drop=True)

    def download_prices_weekly(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        start_buffered = (pd.to_datetime(start_date) - timedelta(days=7)).strftime("%Y-%m-%d")
        
        df_daily = self.download_prices_daily(tickers, start_buffered, end_date)
        df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])

        weekly = (
            df_daily[["Fecha", "Ticker", "Precio_Close", "Volumen_USD"]]
            .set_index("Fecha")
            .groupby("Ticker")[["Precio_Close", "Volumen_USD"]]
            .resample("W-FRI")
            .agg(
                Precio_Close=("Precio_Close", "last"),
                Volumen_USD=("Volumen_USD", "sum"),
            )
        )

        weekly["Precio_Close"] = weekly.groupby("Ticker")["Precio_Close"].ffill()
        weekly = weekly[weekly.index.get_level_values("Fecha") >= pd.to_datetime(start_date)]

        return weekly.reset_index().sort_values(["Ticker", "Fecha"]).reset_index(drop=True)