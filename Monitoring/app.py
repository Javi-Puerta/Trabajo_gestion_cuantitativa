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
    """Parses the ledger to calculate net positions and cost basis (VWAP)."""
    df = pd.read_csv(csv_path)
    
    # Clean strings just in case there are spaces in the CSV
    df['Accion'] = df['Accion'].str.strip().str.upper()
    df['Ticker'] = df['Ticker'].str.strip()
    
    # Assign signs: Positive for COMPRA, Negative for VENTA
    df['Qty_Signed'] = np.where(df['Accion'] == 'COMPRA', df['Cantidad'], -df['Cantidad'])
    
    portfolio = {}
    grouped = df.groupby('Ticker')
    
    for ticker, group in grouped:
        net_qty = group['Qty_Signed'].sum()
        
        # Only track open positions (net_qty != 0)
        if net_qty != 0:
            # Calculate Cost Basis (VWAP). 
            # For a long portfolio, we isolate the COMPRA orders to find average entry cost.
            buys = group[group['Accion'] == 'COMPRA']
            if not buys.empty and net_qty > 0:
                vwap = (buys['Cantidad'] * buys['Precio_Ejecutado']).sum() / buys['Cantidad'].sum()
            else:
                # Fallback for short positions or unusual ledgers
                vwap = group['Precio_Ejecutado'].mean() 
                
            portfolio[ticker] = {
                'cantidad': net_qty,
                'vwap': vwap
            }
            
    return portfolio

# --- 3. STATIC CACHE (Runs Once) ---
@st.cache_data
def load_historical_data(tickers):
    """Downloads 1 year of data to calculate Covariance and Beta."""
    all_symbols = tickers + ['^STOXX50E']
    
    # Download raw data without assuming columns
    raw_data = yf.download(all_symbols, period="1y", progress=False)
    
    # Robust column extraction
    if 'Adj Close' in raw_data.columns:
        hist = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        hist = raw_data['Close']
    else:
        # If both fail, it usually means the tickers aren't recognized
        st.error(f"🚨 YFinance could not find the pricing data. Please verify your tickers have the correct Yahoo suffixes (e.g., SAN.MC, ASML.AS).")
        st.stop()
        
    returns = hist.pct_change().dropna()
    
    # Ensure all required tickers successfully downloaded before doing math
    valid_tickers = [t for t in tickers if t in returns.columns]
    
    cov_matrix = returns[valid_tickers].cov()
    market_var = returns['^STOXX50E'].var()
    betas = {t: returns[t].cov(returns['^STOXX50E']) / market_var for t in valid_tickers}
    
    return cov_matrix, betas

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
    
    # 1. Reconstruct live state from CSV ledger
    try:
        positions = reconstruct_portfolio('../mi_cartera/historial_operaciones.csv')
    except FileNotFoundError:
        st.error("historial_operaciones.csv not found. Please ensure it is in the same directory.")
        st.stop()
        
    tickers = list(positions.keys())
    
    if not tickers:
        st.warning("No open positions found in the ledger.")
        st.stop()
    
    # 2. Load cached quant metrics
    cov_matrix, asset_betas = load_historical_data(tickers)
    
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

    # --- REFRESH LOGIC ---
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Auto-refreshing every {REFRESH_RATE} seconds...")
    time.sleep(REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()