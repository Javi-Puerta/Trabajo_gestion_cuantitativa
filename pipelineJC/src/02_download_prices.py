import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
data_path   = os.path.join(script_dir, '..', 'data')

class PriceDownloader:
    def __init__(self, config_path=config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def download_single_ticker(self, ticker, start_date, end_date):
        """Download OHLCV data for single ticker"""
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,  # or True if you prefer adjusted prices
                threads=False
            )
            if data.empty:
                print(f"No data for {ticker}")
                return None
            data["ticker"] = ticker
            return data
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            return None
    
    def batch_download(self, tickers_list, max_workers=10):
        """Parallel download with error handling"""
        start_date = self.config["date_config"]["start_date"]
        end_date = self.config["date_config"]["end_date"]
        
        output_dir = Path(data_path) / "prices"
        output_dir.mkdir(exist_ok=True)
        
        all_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.download_single_ticker, ticker, start_date, end_date
                ): ticker
                for ticker in tickers_list
            }
            
            for future in futures:
                ticker = futures[future]
                data = future.result()
                if data is not None:
                    all_data.append(data)
                    print(f"Downloaded {ticker}: {len(data)} rows")
        
        if not all_data:
            print("No price data downloaded.")
            return None
        
        combined = pd.concat(all_data)
        combined.index.name = "date"
        
        filepath = output_dir / f"prices_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
        combined.to_parquet(filepath)
        print(f"Saved combined prices to {filepath}")
        return filepath


if __name__ == "__main__":
    # 1) Load constituents from CSV produced by your fetcher
    cons_path = os.path.join(data_path, "constituents", "msci_world_20260206.csv")  # adapt filename
    cons_df = pd.read_csv(cons_path)
    
    # 2) Build ticker list (you may want to filter or deduplicate)
    tickers = sorted(cons_df["ticker"].dropna().unique().tolist())
    print(tickers[:10])  # print first 10 tickers for sanity check
    # For a first test, maybe limit to first 20 tickers
    tickers_test = tickers
    
    downloader = PriceDownloader()
    downloader.batch_download(tickers_test)
