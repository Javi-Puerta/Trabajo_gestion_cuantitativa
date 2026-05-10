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
        Ticker y Fecha. Los precios son no ajustados y semanales (último precio de cada semana).
        """
        pass

class YFinanceProvider(ProveedorDatosBase):
    def download_prices_daily(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
        )

        precios = data["Close"].stack().reset_index()
        precios.columns = ["Fecha", "Ticker", "Precio_Close"]

        volumenes = data["Volume"].stack().reset_index()
        volumenes.columns = ["Fecha", "Ticker", "Volumen"]

        if "Dividends" in data.columns.get_level_values(0):
            dividendos = data["Dividends"].stack().reset_index()
            dividendos.columns = ["Fecha", "Ticker", "Dividendos"]
        else:
            dividendos = precios[["Fecha", "Ticker"]].copy()
            dividendos["Dividendos"] = 0.0

        df = (
            precios
            .merge(volumenes, on=["Fecha", "Ticker"], how="left")
            .merge(dividendos, on=["Fecha", "Ticker"], how="left")
        )

        df["Volumen_USD"] = df["Precio_Close"] * df["Volumen"]
        df["Fecha"] = pd.to_datetime(df["Fecha"])

        fecha_inicio = pd.to_datetime(start_date)
        fecha_fin = pd.to_datetime(end_date)
        todas_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
        todos_tickers = df["Ticker"].unique()

        idx_completo = pd.MultiIndex.from_product(
            [todas_fechas, todos_tickers],
            names=["Fecha", "Ticker"]
        )

        df = df.set_index(["Fecha", "Ticker"]).reindex(idx_completo)

        df["Precio_Close"] = df.groupby("Ticker")["Precio_Close"].ffill()
        df["Volumen_USD"] = df.groupby("Ticker")["Volumen_USD"].ffill()
        df["Dividendos"] = df["Dividendos"].fillna(0.0)

        df = df.reset_index().dropna(subset=["Precio_Close"])

        return df[
            ["Fecha", "Ticker", "Precio_Close", "Volumen_USD", "Dividendos"]
        ].sort_values(["Ticker", "Fecha"]).reset_index(drop=True)

    def download_prices_weekly(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        start_buffered = (pd.to_datetime(start_date) - timedelta(days=7)).strftime("%Y-%m-%d")
        
        df_daily = self.download_prices_daily(tickers, start_buffered, end_date)
        df_daily["Fecha"] = pd.to_datetime(df_daily["Fecha"])

        weekly = (
            df_daily[["Fecha", "Ticker", "Precio_Close", "Volumen_USD", "Dividendos"]]
            .set_index("Fecha")
            .groupby("Ticker")[["Precio_Close", "Volumen_USD", "Dividendos"]]
            .resample("W-FRI")
            .agg(
                Precio_Close=("Precio_Close", "last"),
                Volumen_USD=("Volumen_USD", "sum"),
                Dividendos=("Dividendos", "sum"),
            )
        )

        weekly["Precio_Close"] = weekly.groupby("Ticker")["Precio_Close"].ffill()
        weekly = weekly[weekly.index.get_level_values("Fecha") >= pd.to_datetime(start_date)]

        return weekly.reset_index().sort_values(["Ticker", "Fecha"]).reset_index(drop=True)