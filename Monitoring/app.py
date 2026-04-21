import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Quant Desk Monitor", page_icon="📈", layout="wide")
REFRESH_RATE = 10 
RISK_FREE_RATE = 0.03 
STARTING_NAV = 10_000_000 # The exact Total NAV on March 12, 2026

# --- 2. PORTFOLIO STATE RECONSTRUCTION ---
def reconstruct_portfolio(csv_path):
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, format='mixed')
    inception_date = df['fecha'].min().strftime('%Y-%m-%d')
    
    df['Accion'] = df['Accion'].str.strip().str.upper()
    df['Ticker'] = df['Ticker'].str.strip()
    df['Qty_Signed'] = np.where(df['Accion'] == 'COMPRA', df['Cantidad'], -df['Cantidad'])
    
    total_historical_costs = (df['Cantidad'] * df['CT']).sum()
    
    portfolio = {}
    grouped = df.groupby('Ticker')
    
    for ticker, group in grouped:
        net_qty = group['Qty_Signed'].sum()
        if net_qty != 0:
            buys = group[group['Accion'] == 'COMPRA']
            if not buys.empty and net_qty > 0:
                vwap_net = (buys['Cantidad'] * buys['Precio_Ejecutado']).sum() / buys['Cantidad'].sum()
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

# --- 2.5 TRUE HISTORICAL RETURN VECTOR ---
def calculate_performance_vectors(csv_path, historical_prices, index_prices, starting_nav):
    """Calculates daily return vectors for the strategy and index to build a Base 100 chart."""
    df = pd.read_csv(csv_path)
    # 1. Read dates with European format
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, format='mixed').dt.normalize()
    
    df['Accion'] = df['Accion'].str.strip().str.upper()
    df['Ticker'] = df['Ticker'].str.strip()
    
    # 2. Track Cash Flows using Precio_Ejecutado (Price + Costs)
    df['Cash_Impact'] = np.where(df['Accion'] == 'COMPRA', 
                                 -df['Cantidad'] * df['Precio_Ejecutado'], 
                                 df['Cantidad'] * df['Precio_Ejecutado'])
    df['Qty_Signed'] = np.where(df['Accion'] == 'COMPRA', df['Cantidad'], -df['Cantidad'])
    
    daily_trades = df.groupby(['fecha', 'Ticker'])['Qty_Signed'].sum().unstack(fill_value=0)
    daily_cash = df.groupby('fecha')['Cash_Impact'].sum()
    
    all_dates = historical_prices.index.normalize().unique()
    
    # 3. Build cumulative running balances
    positions_history = daily_trades.reindex(all_dates, fill_value=0).cumsum()
    cash_history = starting_nav + daily_cash.reindex(all_dates, fill_value=0).cumsum()
    
    valid_tickers = [t for t in positions_history.columns if t in historical_prices.columns]
    positions_history = positions_history[valid_tickers]
    prices_aligned = historical_prices[valid_tickers]
    
    # 4. Calculate Absolute Total NAV (Cash + Market Data from YFinance)
    invested_value = (positions_history * prices_aligned).sum(axis=1)
    true_nav_history = invested_value + cash_history
    
    # 5. GENERATE THE RETURN VECTORS
    # Calculate the daily percentage change of the True NAV
    strategy_daily_returns = true_nav_history.pct_change().fillna(0)
    # Calculate the daily percentage change of the Index
    index_daily_returns = index_prices.pct_change().fillna(0)
    
    # 6. CALCULATE BASE 100
    strategy_base_100 = 100 * (1 + strategy_daily_returns).cumprod()
    index_base_100 = 100 * (1 + index_daily_returns).cumprod()
    
    return strategy_base_100, index_base_100

# --- 3. STATIC CACHE ---
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
    """Fetches spot prices, with a robust fallback for weekends/closed markets."""
    # 1. Try to get live 1-minute intraday data
    raw_data = yf.download(tickers, period="1d", interval="1m", progress=False)
    
    # 2. FALLBACK: If the market is closed and 1m data is empty, get the last 5 days of daily data
    if raw_data.empty:
        raw_data = yf.download(tickers, period="5d", interval="1d", progress=False)
        
    # 3. Extract the correct pricing column
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        # Ultimate fallback if Yahoo completely fails
        return {t: 0 for t in tickers} 

    # 4. Final safety check to prevent the IndexError
    if data.empty:
        return {t: 0 for t in tickers}

    # 5. Return the prices
    if isinstance(data, pd.Series):
        return {tickers[0]: data.iloc[-1]}
        
    # ffill() carries the last traded price forward in case a specific stock is halted
    return data.ffill().iloc[-1].to_dict()

# --- 5. MAIN DASHBOARD LOOP ---
def main():
    st.title("🇪🇺 Eurostoxx50 Quant Monitor")
    
    try:
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
        cost_basis_net = qty * vwap_net
        cost_basis_gross = qty * vwap_gross
        
        unrealized_net_pnl = position_value - cost_basis_net
        unrealized_gross_pnl = position_value - cost_basis_gross
        open_position_costs = cost_basis_net - cost_basis_gross 
        
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
    
# Calculate Advanced Risk & Performance Metrics
    if total_nav > 0:
        df_portfolio['Weight'] = df_portfolio['Notional (€)'] / total_nav
        weights = df_portfolio['Weight'].values
        
        # 1. Risk Metrics
        live_beta = sum(row['Weight'] * asset_betas[row['Ticker']] for _, row in df_portfolio.iterrows())
        port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        port_std_dev = np.sqrt(port_variance)
        live_var_99 = total_nav * port_std_dev * 2.33 
        
        # 2. Backcasted Performance Metrics (CUMULATIVE & DAILY - No Annualization)
        asset_returns = historical_returns[tickers]
        portfolio_daily_returns = asset_returns.dot(weights)
        index_daily_returns = historical_returns['^STOXX50E']
        
        # Calculate Total Cumulative Returns
        port_total_ret = (1 + portfolio_daily_returns).prod() - 1
        idx_total_ret = (1 + index_daily_returns).prod() - 1
        
        # Adjust Risk-Free Rate for the exact number of trading days
        n_days = len(portfolio_daily_returns)
        daily_rf = RISK_FREE_RATE / 252
        period_rf = (1 + daily_rf)**n_days - 1
        
        # Daily Sharpe Ratio
        port_daily_mean = portfolio_daily_returns.mean()
        port_daily_std = portfolio_daily_returns.std()
        daily_sharpe = (port_daily_mean - daily_rf) / port_daily_std if port_daily_std > 0 else 0
        
        # Cumulative Jensen's Alpha (Total exact alpha since inception)
        jensens_alpha_cum = port_total_ret - (period_rf + live_beta * (idx_total_ret - period_rf))
        
    else:
        live_beta, live_var_99, daily_sharpe, jensens_alpha_cum = 0, 0, 0, 0
        portfolio_daily_returns, index_daily_returns = pd.Series(), pd.Series()

    # --- RENDER UI ---
    st.markdown("### 📊 Live Portfolio State")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total NAV", value=f"€{total_nav:,.2f}")
    with col2:
        pnl_color = "normal" if total_unrealized_net >= 0 else "inverse"
        st.metric(label="Real Net PnL", value=f"€{total_unrealized_net:,.2f}", delta=f"€{total_unrealized_net:,.2f}", delta_color=pnl_color)
    with col3:
        st.metric("Unrealized Gross PnL", f"€{total_unrealized_gross:,.2f}")
    with col4:
        st.metric("Open Position Friction", f"€{-total_open_costs:,.2f}")

    st.markdown("### 📐 Performance & Risk Metrics")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Live Portfolio Beta", f"{live_beta:.2f}")
    r2.metric("1-Day 99% VaR", f"€{-live_var_99:,.2f}")
    r3.metric("Daily Sharpe Ratio", f"{daily_sharpe:.3f}")
    r4.metric("Cumulative Alpha", f"{jensens_alpha_cum * 100:.2f}%", help=f"Total absolute alpha generated over the {n_days}-day trading period.")
        
    st.markdown("---")
    
    # Portfolio Table
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

# --- CHARTS SECTION (SIDE-BY-SIDE) ---
    st.markdown("---")
    
    col_chart, col_pie = st.columns([7, 3])
    
    with col_chart:
        st.subheader("📈 Strategy Performance vs Benchmark (Base 100)")
        if total_nav > 0:
            
            raw_data = yf.download(tickers + ['^STOXX50E'], start=inception_date, progress=False)
            hist_prices = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns else raw_data['Close']
            
            index_prices = hist_prices['^STOXX50E']
            # Drop the index from the portfolio prices
            portfolio_prices = hist_prices.drop(columns=['^STOXX50E'], errors='ignore') 
            
            # Execute your logic to get the two Base 100 vectors
            strategy_b100, index_b100 = calculate_performance_vectors(
                '../mi_cartera/historial_operaciones.csv', 
                portfolio_prices,
                index_prices,
                STARTING_NAV # Using your 10M starting point
            )
            
            # Plot the vectors
            df_chart = pd.DataFrame({
                'Strategy (True NAV)': strategy_b100,
                'Eurostoxx 50': index_b100
            })
            
            st.line_chart(df_chart, color=["#1f77b4", "#7f7f7f"])
            
    with col_pie:
        st.subheader("🍩 Capital Allocation (Invested)")
        if total_nav > 0:
            fig = px.pie(
                df_portfolio, 
                values='Notional (€)', 
                names='Ticker', 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # --- REFRESH LOGIC ---
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Total Historical Costs: €{total_historical_costs:,.2f} | Auto-refreshing every {REFRESH_RATE} seconds...")
    time.sleep(REFRESH_RATE)
    st.rerun()

if __name__ == "__main__":
    main()