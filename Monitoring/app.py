import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Quant Desk Monitor", page_icon="📈", layout="wide")
REFRESH_RATE = 10 # Seconds

# --- 2. PORTFOLIO STATE RECONSTRUCTION ---
def reconstruct_portfolio(csv_path):
    """Parses the ledger to calculate net positions, VWAP, and the inception date."""
    df = pd.read_csv(csv_path)
    
    # Parse the date column (dayfirst=True handles European DD/MM/YYYY formats properly)
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, format='mixed')
    inception_date = df['fecha'].min().strftime('%Y-%m-%d')
    
    df['Accion'] = df['Accion'].str.strip().str.upper()
    df['Ticker'] = df['Ticker'].str.strip()
    df['Qty_Signed'] = np.where(df['Accion'] == 'COMPRA', df['Cantidad'], -df['Cantidad'])
    
    portfolio = {}
    grouped = df.groupby('Ticker')
    
    for ticker, group in grouped:
        net_qty = group['Qty_Signed'].sum()
        if net_qty != 0:
            buys = group[group['Accion'] == 'COMPRA']
            if not buys.empty and net_qty > 0:
                vwap = (buys['Cantidad'] * buys['Precio_Ejecutado']).sum() / buys['Cantidad'].sum()
            else:
                vwap = group['Precio_Ejecutado'].mean() 
                
            portfolio[ticker] = {
                'cantidad': net_qty,
                'vwap': vwap
            }
            
    # Now returning TWO things: the dictionary and the start date
    return portfolio, inception_date

# --- 3. STATIC CACHE (Runs Once) ---
@st.cache_data
def load_historical_data(tickers, start_date):
    """Downloads data from inception to calculate Covariance, Beta, and Returns."""
    all_symbols = tickers + ['^STOXX50E']
    
    # Fetch from inception date instead of 1y
    raw_data = yf.download(all_symbols, start=start_date, progress=False)
    
    if 'Adj Close' in raw_data.columns:
        hist = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        hist = raw_data['Close']
    else:
        st.error(f"🚨 YFinance could not find the pricing data.")
        st.stop()
        
    returns = hist.pct_change().dropna()
    valid_tickers = [t for t in tickers if t in returns.columns]
    
    cov_matrix = returns[valid_tickers].cov()
    market_var = returns['^STOXX50E'].var()
    betas = {t: returns[t].cov(returns['^STOXX50E']) / market_var for t in valid_tickers}
    
    return cov_matrix, betas, returns

# --- 4. LIVE DATA FETCHING ---
def fetch_live_prices(tickers):
    """Fetches the most recent spot price."""
    raw_data = yf.download(tickers, period="1d", interval="1m", progress=False)
    
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        return {} # Return empty dict if no data found

    if isinstance(data, pd.Series):
        return {tickers[0]: data.iloc[-1]}
    
    # .dropna() ensures we only get the latest price for stocks that are currently trading
    return data.ffill().iloc[-1].to_dict()

# --- 5. MAIN DASHBOARD LOOP ---
def main():
    st.title("🇪🇺 Eurostoxx50 Quant Monitor")
    
    # 1. Reconstruct live state from CSV ledger AND catch the inception date
    try:
        positions, inception_date = reconstruct_portfolio('../mi_cartera/historial_operaciones.csv') # Or uploaded_csv if you used the uploader
    except FileNotFoundError:
        st.error("historial_operaciones.csv not found.")
        st.stop()
        
    tickers = list(positions.keys())
    
    if not tickers:
        st.warning("No open positions found in the ledger.")
        st.stop()
    
    # 2. Load cached quant metrics passing the inception date
    cov_matrix, asset_betas, historical_returns = load_historical_data(tickers, inception_date)
    
    # 3. Fetch live spot prices
    live_prices = fetch_live_prices(tickers)
    
    # --- CALCULATE LIVE METRICS ---
    total_nav = 0
    total_unrealized_pnl = 0
    portfolio_data = []
    
    for ticker, data in positions.items():
        qty = data['cantidad']
        vwap = data['vwap']
        spot_price = live_prices.get(ticker, 0)
        
        position_value = qty * spot_price
        cost_basis_total = qty * vwap
        unrealized_pnl = position_value - cost_basis_total
        
        total_nav += position_value
        total_unrealized_pnl += unrealized_pnl
        
        portfolio_data.append({
            "Ticker": ticker,
            "Net Qty": qty,
            "Avg Entry (VWAP)": round(vwap, 2),
            "Spot Price (€)": round(spot_price, 2),
            "Unrealized PnL (€)": round(unrealized_pnl, 2),
            "Notional (€)": round(position_value, 2)
        })
        
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # Live Portfolio Beta & VaR
    if total_nav > 0:
        df_portfolio['Weight'] = df_portfolio['Notional (€)'] / total_nav
        weights = df_portfolio['Weight'].values
        
        live_beta = sum(row['Weight'] * asset_betas[row['Ticker']] for _, row in df_portfolio.iterrows())
        
        port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        port_std_dev = np.sqrt(port_variance)
        
        z_score = 2.33 
        live_var_99 = total_nav * port_std_dev * z_score
    else:
        live_beta, live_var_99 = 0, 0

    # --- RENDER UI ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total NAV", value=f"€{total_nav:,.2f}")
    with col2:
        # Green if profitable, red if negative
        pnl_color = "normal" if total_unrealized_pnl >= 0 else "inverse"
        st.metric(label="Unrealized PnL", value=f"€{total_unrealized_pnl:,.2f}", delta=f"€{total_unrealized_pnl:,.2f}", delta_color=pnl_color)
    with col3:
        st.metric(label="Live Portfolio Beta", value=f"{live_beta:.2f}")
    with col4:
        st.metric(label="1-Day 99% VaR", value=f"€{-live_var_99:,.2f}")
        
    st.markdown("---")
    
    st.subheader("Live Portfolio Ledger")
    
    # Apply a color gradient to the PnL column for better readability
    def color_pnl(val):
        color = 'rgba(39, 174, 96, 0.3)' if val > 0 else 'rgba(231, 76, 60, 0.3)' if val < 0 else ''
        return f'background-color: {color}'

    styled_df = df_portfolio.style.format({
        "Avg Entry (VWAP)": "€{:.2f}",
        "Spot Price (€)": "€{:.2f}",
        "Unrealized PnL (€)": "€{:,.2f}",
        "Notional (€)": "€{:,.2f}",
        "Weight": "{:.2%}"
    }).map(color_pnl, subset=['Unrealized PnL (€)'])
    
    st.dataframe(styled_df, use_container_width=True)

    # --- HISTORICAL PERFORMANCE CHART ---
    st.markdown("---")
    st.subheader("📈 1-Year Backcasted Performance vs Benchmark (Base 100)")
    
    if total_nav > 0:
        # 1. Isolate the historical returns of just our portfolio assets
        asset_returns = historical_returns[tickers]
        
        # 2. Multiply daily returns by current weights (Dot Product)
        # This gives us a single Pandas Series representing the daily return of the portfolio
        portfolio_daily_returns = asset_returns.dot(weights)
        
        # 3. Isolate the benchmark returns
        index_daily_returns = historical_returns['^STOXX50E']
        
        # 4. Calculate Cumulative Returns starting at 100
        # (1 + r).cumprod() gives the growth factor over time
        df_chart = pd.DataFrame({
            'Current Portfolio': (1 + portfolio_daily_returns).cumprod() * 100,
            'Eurostoxx 50 (^STOXX50E)': (1 + index_daily_returns).cumprod() * 100
        })
        
        # 5. Native Streamlit Line Chart
        # We define custom colors: Blue for the Portfolio, Gray for the Index
        st.line_chart(df_chart, color=["#1f77b4", "#7f7f7f"])
    else:
        st.info("Portfolio must have a positive NAV to render the performance chart.")

    # --- REFRESH LOGIC ---
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Auto-refreshing every {REFRESH_RATE} seconds...")
    time.sleep(REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()