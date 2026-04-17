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
    """Parses the ledger to calculate net positions, Gross VWAP, Net VWAP, and inception."""
    df = pd.read_csv(csv_path)
    
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, format='mixed')
    inception_date = df['fecha'].min().strftime('%Y-%m-%d')
    
    df['Accion'] = df['Accion'].str.strip().str.upper()
    df['Ticker'] = df['Ticker'].str.strip()
    df['Qty_Signed'] = np.where(df['Accion'] == 'COMPRA', df['Cantidad'], -df['Cantidad'])
    
    # NEW: Calculate Total Strategy Friction (Costs of ALL trades, open and closed)
    total_historical_costs = (df['Cantidad'] * df['CT']).sum()
    
    portfolio = {}
    grouped = df.groupby('Ticker')
    
    for ticker, group in grouped:
        net_qty = group['Qty_Signed'].sum()
        if net_qty != 0:
            buys = group[group['Accion'] == 'COMPRA']
            if not buys.empty and net_qty > 0:
                # NET VWAP uses the executed price (includes costs)
                vwap_net = (buys['Cantidad'] * buys['Precio_Ejecutado']).sum() / buys['Cantidad'].sum()
                # GROSS VWAP uses the clean market price (excludes costs)
                vwap_gross = (buys['Cantidad'] * buys['Precio']).sum() / buys['Cantidad'].sum()
            else:
                vwap_net = group['Precio_Ejecutado'].mean() 
                vwap_gross = group['Precio'].mean()
                
            portfolio[ticker] = {
                'cantidad': net_qty,
                'vwap_net': vwap_net,
                'vwap_gross': vwap_gross
            }
            
    return portfolio, inception_date, total_historical_costs

# --- 3. STATIC CACHE (Runs Once) ---
@st.cache_data
def load_historical_data(tickers, start_date):
    all_symbols = tickers + ['^STOXX50E']
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
    raw_data = yf.download(tickers, period="1d", interval="1m", progress=False)
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        return {}
    if isinstance(data, pd.Series):
        return {tickers[0]: data.iloc[-1]}
    return data.ffill().iloc[-1].to_dict() 

# --- 5. MAIN DASHBOARD LOOP ---
def main():
    st.title("🇪🇺 Eurostoxx50 Quant Monitor")
    
    try:
        # Now catching three variables, including the total historical costs
        positions, inception_date, total_historical_costs = reconstruct_portfolio('../mi_cartera/historial_operaciones.csv') 
    except FileNotFoundError:
        st.error("historial_operaciones.csv not found.")
        st.stop()
        
    tickers = list(positions.keys())
    if not tickers:
        st.warning("No open positions found in the ledger.")
        st.stop()
    
    cov_matrix, asset_betas, historical_returns = load_historical_data(tickers, inception_date)
    live_prices = fetch_live_prices(tickers)
    
    # --- CALCULATE LIVE METRICS ---
    total_nav = 0
    total_unrealized_gross = 0
    total_unrealized_net = 0
    total_open_costs = 0
    portfolio_data = []
    
    for ticker, data in positions.items():
        qty = data['cantidad']
        vwap_net = data['vwap_net']
        vwap_gross = data['vwap_gross']
        spot_price = live_prices.get(ticker, 0)
        
        position_value = qty * spot_price
        
        # Calculate Net vs Gross Basis
        cost_basis_net = qty * vwap_net
        cost_basis_gross = qty * vwap_gross
        
        # Decompose the PnL
        unrealized_net_pnl = position_value - cost_basis_net
        unrealized_gross_pnl = position_value - cost_basis_gross
        open_position_costs = cost_basis_net - cost_basis_gross # The CT paid for these open lots
        
        total_nav += position_value
        total_unrealized_net += unrealized_net_pnl
        total_unrealized_gross += unrealized_gross_pnl
        total_open_costs += open_position_costs
        
        portfolio_data.append({
            "Ticker": ticker,
            "Net Qty": qty,
            "Spot Price (€)": round(spot_price, 2),
            "Gross PnL (€)": round(unrealized_gross_pnl, 2),
            "Costs Paid (€)": round(open_position_costs, 2),
            "Real Net PnL (€)": round(unrealized_net_pnl, 2),
            "Notional (€)": round(position_value, 2)
        })
        
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # Calculate Risk Metrics
    if total_nav > 0:
        df_portfolio['Weight'] = df_portfolio['Notional (€)'] / total_nav
        weights = df_portfolio['Weight'].values
        live_beta = sum(row['Weight'] * asset_betas[row['Ticker']] for _, row in df_portfolio.iterrows())
        port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        port_std_dev = np.sqrt(port_variance)
        live_var_99 = total_nav * port_std_dev * 2.33 
    else:
        live_beta, live_var_99 = 0, 0

    # --- RENDER UI ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total NAV", value=f"€{total_nav:,.2f}")
    with col2:
        pnl_color = "normal" if total_unrealized_net >= 0 else "inverse"
        st.metric(label="Real Net PnL", value=f"€{total_unrealized_net:,.2f}", delta=f"€{total_unrealized_net:,.2f}", delta_color=pnl_color)
    with col3:
        st.metric(label="Live Portfolio Beta", value=f"{live_beta:.2f}")
    with col4:
        st.metric(label="1-Day 99% VaR", value=f"€{-live_var_99:,.2f}")

    # NEW: PnL Decomposition HUD
    st.markdown("### 💸 Strategy Friction & PnL Decomposition")
    st.info(f"**Total Lifetime Strategy Costs:** €{total_historical_costs:,.2f} (Includes all closed and open trades since inception)")
    
    dec1, dec2, dec3 = st.columns(3)
    dec1.metric("Unrealized Gross PnL", f"€{total_unrealized_gross:,.2f}", help="Market profit ignoring transaction costs.")
    dec2.metric("Open Position Friction", f"€{-total_open_costs:,.2f}", help="Transaction costs paid solely for currently open lots.")
    dec3.metric("Unrealized Net PnL", f"€{total_unrealized_net:,.2f}", help="Real profit remaining after costs.")
        
    st.markdown("---")
    st.subheader("Live Portfolio Ledger")
    
    def color_pnl(val):
        return f"background-color: {'rgba(39, 174, 96, 0.3)' if val > 0 else 'rgba(231, 76, 60, 0.3)' if val < 0 else ''}"

    styled_df = df_portfolio.style.format({
        "Spot Price (€)": "€{:.2f}",
        "Gross PnL (€)": "€{:,.2f}",
        "Costs Paid (€)": "€{:,.2f}",
        "Real Net PnL (€)": "€{:,.2f}",
        "Notional (€)": "€{:,.2f}",
        "Weight": "{:.2%}"
    }).map(color_pnl, subset=['Real Net PnL (€)', 'Gross PnL (€)'])
    
    st.dataframe(styled_df, use_container_width=True)

    # --- HISTORICAL PERFORMANCE CHART ---
    st.markdown("---")
    st.subheader(f"📈 Backcasted Performance vs Benchmark (Since {inception_date})")
    
    if total_nav > 0:
        asset_returns = historical_returns[tickers]
        portfolio_daily_returns = asset_returns.dot(weights)
        index_daily_returns = historical_returns['^STOXX50E']
        
        df_chart = pd.DataFrame({
            'Current Portfolio': (1 + portfolio_daily_returns).cumprod() * 100,
            'Eurostoxx 50 (^STOXX50E)': (1 + index_daily_returns).cumprod() * 100
        })
        st.line_chart(df_chart, color=["#1f77b4", "#7f7f7f"])
    
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Auto-refreshing every {REFRESH_RATE} seconds...")
    time.sleep(REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()