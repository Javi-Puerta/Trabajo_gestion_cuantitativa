import pandas as pd
import requests
import yaml
from pathlib import Path
from datetime import datetime
import os
from bs4 import BeautifulSoup
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
data_path   = os.path.join(script_dir, '..', 'data')



def clean_company_name(name: str) -> str:
    """
    Simplify company name for Yahoo search:
    - remove accents
    - remove parentheses and content
    - remove "THE" at start/end
    - remove legal suffixes
    - trim whitespace
    """
    if not isinstance(name, str):
        return ""


    # Remove leading/trailing whitespace
    name = name.strip()


    # Normalize accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))


    # Remove parentheses and everything inside
    name = re.sub(r"\([^)]*\)", "", name)


    # Remove "THE" at beginning or end
    name = re.sub(r"^THE\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+THE$", "", name, flags=re.IGNORECASE)


    # Remove common legal suffixes
    legal_suffixes = [
        r"\s+AG$", r"\s+S\.P\.A\.$", r"\s+N\.V\.$", r"\s+PLC$",
        r"\s+OYJ$", r"\s+NV/SA$", r"\s+CORPORATION$", r"\s+INCORPORATED$",
        r"\s+SE$", r"\s+S\.A\.$", r"\s+LTD\.?$", r"\s+LIMITED$",
        r"\s+INC\.$", r"\s+CORP\.$", r"\s+GROUP$", r"\s+CO\.$"
    ]
    for suffix in legal_suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)


    # Remove stock class letters at end (A, B, etc.)
    name = re.sub(r"\s+[A-Z]$", "", name)


    # Remove apostrophes
    name = name.replace("'", "").replace("'", "").replace("'", "").replace("`", "").replace(",", "")


    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()


    return name



@lru_cache(maxsize=2000)
def cached_clean_company_name(name: str) -> str:
    """Cached version of clean_company_name to avoid reprocessing."""
    return clean_company_name(name)



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        requests.RequestException,
        requests.ConnectionError,
        requests.Timeout
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def name_to_yahoo_ticker(company_name: str, session: requests.Session = None) -> str | None:
    """
    Given a company name, return the most likely Yahoo Finance ticker.

    Logic:
    - Clean the name (accents, legal suffixes, etc.)
    - Call Yahoo search API
    - Prefer quoteType == 'EQUITY'
    - Among candidates, rank by:
        * Yahoo's own 'score'
        * Preferred exchanges (to avoid wrong listing like Milan vs Madrid)
    - If there are no EQUITY quotes at all, fall back to the best overall quote.
    - Retries on network errors with exponential backoff.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    clean_name = cached_clean_company_name(company_name)

    params = {
        "q": clean_name,
        "quotes_count": 10,  # get several candidates
        "news_count": 0
    }

    # Use provided session or create a new request
    request_func = session.get if session else requests.get

    res = request_func(url, headers=headers, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()
    quotes = data.get("quotes", [])
    if not quotes:
        return None

    # 1) Helper: pick best quote from a list, using score + preferred exchange
    PREFERRED_EXCHANGES = [
        # Iberia / Spain first to fix Caixabank-like cases
        "MCE", "BME",        # Spain (Bolsa de Madrid / Mercado Continuo)
        # Major US exchanges
        "NYQ", "NMS", "NCM", "NGM",
        # UK
        "LSE", "LSEI",
        # Germany
        "XETRA", "GER",
        # France / Italy
        "PAR", "MIL",
        # Japan
        "TSE", "JPX",
    ]
    exch_rank = {exch: i for i, exch in enumerate(PREFERRED_EXCHANGES)}

    def best_quote(candidates):
        if not candidates:
            return None
        best = None
        best_score = float("-inf")
        for q in candidates:
            base = q.get("score", 0.0) or 0.0
            exch = (q.get("exchange") or "").upper()
            # Higher weight for preferred exchanges (lower index = higher bonus)
            bonus = 0.0
            if exch in exch_rank:
                bonus += 10.0 * (len(PREFERRED_EXCHANGES) - exch_rank[exch])
            total = base + bonus
            if total > best_score:
                best_score = total
                best = q
        return best

    # 2) Prefer EQUITY quotes, if any
    equity_quotes = [q for q in quotes if q.get("quoteType") == "EQUITY"]

    if equity_quotes:
        chosen = best_quote(equity_quotes)
    else:
        # Fallback: no equity at all, pick best from all quotes
        chosen = best_quote(quotes)

    if not chosen:
        return None

    ticker = chosen.get("symbol")
    quote_type = chosen.get("quoteType")
    exchange = chosen.get("exchange", "")
    logger.debug(f"{company_name} -> {ticker} (type: {quote_type}, exchange: {exchange})")

    return ticker


def fetch_ticker_batch(
    companies: list[str], 
    max_workers: int = 20,
    rate_limit_delay: float = 0.0
) -> dict[str, str | None]:
    """
    Fetch tickers for a batch of companies in parallel using threading.
    
    Args:
        companies: List of company names
        max_workers: Number of concurrent threads (default 20)
        rate_limit_delay: Optional delay between requests in seconds (default 0.0)
        
    Returns:
        Dictionary mapping company names to tickers
        
    Performance:
    - Uses connection pooling with increased pool size
    - Automatic retry with exponential backoff
    - Caching to avoid duplicate lookups
    - Filters for EQUITY only
    """
    results = {}
    
    # Create a session with optimized connection pooling
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=0  # We handle retries with tenacity
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    def fetch_single(company_name: str) -> tuple[str, str | None]:
        """Fetch ticker for a single company with optional rate limiting."""
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        
        try:
            ticker = name_to_yahoo_ticker(company_name, session)
            return (company_name, ticker)
        except Exception as e:
            logger.error(f"Failed after all retries for '{company_name}': {e}")
            return (company_name, None)
    
    # Use ThreadPoolExecutor for parallel requests
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_company = {
            executor.submit(fetch_single, company): company 
            for company in companies
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_company), 1):
            company = future_to_company[future]
            try:
                company_name, ticker = future.result()
                results[company_name] = ticker
                
                # Progress indicator with time estimate
                if i % 50 == 0 or i == len(companies):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(companies) - i) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {i}/{len(companies)} ({i/len(companies)*100:.1f}%) | "
                        f"Rate: {rate:.1f} req/s | ETA: {eta:.0f}s"
                    )
            except Exception as e:
                logger.error(f"Error processing {company}: {e}")
                results[company] = None
    
    session.close()
    elapsed_total = time.time() - start_time
    logger.info(f"Completed {len(companies)} lookups in {elapsed_total:.1f}s ({len(companies)/elapsed_total:.1f} req/s)")
    return results



class ConstituentsFetcher:
    def __init__(self, config_path=config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path(data_path) / "constituents"
        self.output_dir.mkdir(exist_ok=True)
    
    def fetch_msci_constituents_web(
        self, 
        max_workers: int = 20,
        rate_limit_delay: float = 0.0,
        use_cache: bool = True,
        cache_file: str = None
    ):
        """
        Web scrape ALL MSCI World constituents from all pages.
        Parallel ticker lookup for significant speedup.
        
        Args:
            max_workers: Number of concurrent threads for ticker lookups (default 20)
            rate_limit_delay: Delay between requests in seconds (0.0 = no delay)
            use_cache: Whether to use cached company names if available
            cache_file: Path to cached CSV with company names
            
        Performance optimizations:
        - Two-phase approach: scrape names (fast), then lookup tickers (parallel)
        - Connection pooling with increased pool size
        - Exponential backoff retry logic
        - Optional caching to avoid re-scraping
        - EQUITY-only filtering to avoid funds/ETFs
        """
        base_url = "https://www.marketscreener.com/quote/index/MSCI-WORLD-107361487/components/"
        headers = {"User-Agent": "Mozilla/5.0"}


        all_rows = []
        
        # Check cache first
        if use_cache and cache_file and Path(cache_file).exists():
            logger.info(f"Loading company names from cache: {cache_file}")
            df_cache = pd.read_csv(cache_file)
            all_rows = df_cache[['company_name']].to_dict('records')
            for row in all_rows:
                row['index'] = 'MSCI World'
                row['ticker'] = None
            logger.info(f"Loaded {len(all_rows)} companies from cache")
        else:
            # First pass: collect all company names (fast, sequential scraping)
            logger.info("Phase 1: Scraping company names from all pages...")
            page = 1
            max_pages = 100  # hard safety limit


            while page <= max_pages:
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}?p={page}"


                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.raise_for_status()
                except Exception as e:
                    logger.error(f"Error fetching page {page}: {e}")
                    break


                soup = BeautifulSoup(resp.text, "lxml")


                table = soup.find("table", id="stocks_table")
                if table is None:
                    logger.info(f"No table on page {page}, stopping.")
                    break


                rows = table.find_all("tr")[1:]  # skip header
                if not rows:
                    logger.info(f"No rows on page {page}, stopping.")
                    break


                page_rows_before = len(all_rows)


                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) < 2:
                        continue


                    name = cols[1].get_text(strip=True)
                    all_rows.append({
                        "company_name": name,
                        "ticker": None,  # Will be filled in next phase
                        "index": "MSCI World"
                    })


                page_rows_after = len(all_rows)
                added = page_rows_after - page_rows_before
                logger.info(f"Scraped page {page}, added {added} rows, total so far: {len(all_rows)}")


                if added == 0:
                    logger.info(f"No new rows on page {page}, stopping.")
                    break


                page += 1
                time.sleep(0.5)  # Be nice to the server


            logger.info(f"Phase 1 complete: Found {len(all_rows)} companies")


        # Second pass: parallel ticker lookups (this is the slow part we optimize)
        if all_rows:
            logger.info(f"\nPhase 2: Fetching tickers in parallel (max_workers={max_workers}, rate_limit={rate_limit_delay}s)...")
            company_names = [row["company_name"] for row in all_rows]
            
            ticker_map = fetch_ticker_batch(
                company_names, 
                max_workers=max_workers,
                rate_limit_delay=rate_limit_delay
            )
            
            # Update the rows with fetched tickers
            for row in all_rows:
                row["ticker"] = ticker_map.get(row["company_name"])
            
            logger.info("Phase 2 complete: Ticker lookup finished")


        df = pd.DataFrame(all_rows)
        return df


    def save_constituents(self, df, index_name):
        """Save constituents to CSV with timestamp."""
        filename = f"{index_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} constituents to {filepath}")
        return filepath



# Usage examples
if __name__ == "__main__":
    fetcher = ConstituentsFetcher()
    
    # ==============================================================
    # OPTION 1: Maximum speed (recommended for most users)
    # ==============================================================
    # - 20 concurrent workers
    # - No rate limiting
    # - ~3-5 minutes for 1500 companies
    # - Stays well under Yahoo's rate limits (~2000 req/hour)
    
    world_df = fetcher.fetch_msci_constituents_web(
    max_workers=20,
    rate_limit_delay=0.0)
    
    # ==============================================================
    # OPTION 2: Conservative (if you hit rate limits)
    # ==============================================================
    # - 10 concurrent workers
    # - 0.2s delay between requests = ~300 req/min = ~18,000 req/hour
    # - ~5-8 minutes for 1500 companies
    
    # world_df = fetcher.fetch_msci_constituents_web(
    #     max_workers=10,
    #     rate_limit_delay=0.2
    # )
    
    # ==============================================================
    # OPTION 3: Use cached company names (fastest for re-runs)
    # ==============================================================
    # - Skip web scraping if you already have company names
    # - Only fetch missing tickers
    # - Useful for incremental updates
    
    # world_df = fetcher.fetch_msci_constituents_web(
    #     max_workers=20,
    #     use_cache=True,
    #     cache_file=Path(data_path) / "constituents" / "msci_world_20260205.csv"
    # )
    
    def apply_manual_overrides(df, overrides):
        df = df.copy()
        df["ticker"] = df.apply(
        lambda row: overrides.get(row["company_name"], row["ticker"]),
        axis=1
        )
        return df

    manual_overrides = {
    "DEUTSCHE BÖRSE AG":                    "DB1.DE",        # Xetra
    "DAIKIN INDUSTRIES,LTD.":               "6367.T",        # Tokyo
    "COCA-COLA HELLENIC":                   "CCH.L",         # Coca‑Cola HBC AG, London
    "JULIUS BÄR GRUPPE AG":                 "BAER.SW",       # SIX Swiss
    "ASPEN TECHNOLOGY, INC.":               "AZPN",          # Nasdaq
    "MARATHON OIL CORPORATION":             "MRO",           # NYSE
    "TOKYO GAS CO.,LTD.":                   "9531.T",        # Tokyo
    "JUNIPER NETWORKS, INC.":               "JNPR",          # NYSE
    "THE JM SMUCKER COMPANY":               "SJM",           # NYSE
    "OBIC CO.,LTD.":                        "4684.T",        # Tokyo
    "WALGREENS BOOTS ALLIANCE, INC.":       "WBA",           # Nasdaq
    "SHIZUOKA FINANCIAL GROUP,INC.":        "5831.T",        # Tokyo
    "THE INTERPUBLIC GROUP OF COMPANIES, INC.": "IPG",       # NYSE
    "VOLVO CARS":                           "VOLCAR-B.ST",   # Stockholm
    "HARGREAVES LANSDOWN":                  "HL.L",          # London
    "DAITO TRUST CONSTRUCTION CO.,LTD.":    "1878.T",        # Tokyo
    "NISSIN FOODS HOLDINGS CO.,LTD.":       "2897.T",        # Tokyo
    "SG HOLDINGS CO.,LTD.":                 "9143.T",        # Tokyo,

    # Also fix Caixabank primary listing:
    # Adjust the key to match exactly the company_name in your DataFrame
    "CAIXABANK SA":                         "CABK.MC",
    "CAIXABANK, S.A.":                      "CABK.MC",}

    world_df = apply_manual_overrides(world_df, manual_overrides)


    # ==============================================================
    # Results summary
    # ==============================================================
    missing_count = world_df["ticker"].isna().sum()
    success_rate = (len(world_df) - missing_count) / len(world_df) * 100
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total companies: {len(world_df)}")
    print(f"Tickers found: {len(world_df) - missing_count}")
    print(f"Missing tickers: {missing_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if missing_count > 0:
        print(f"\nCompanies with missing tickers:")
        print(world_df[world_df["ticker"].isna()]["company_name"].to_string(index=False))
    
    # Save results
    world_path = fetcher.save_constituents(world_df, "MSCI World")
    print(f"\nSaved to: {world_path}")
