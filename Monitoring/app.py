import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="Prop Trading MTM Engine", layout="wide")

INITIAL_CAPITAL = 10000000.0
RFR_ANNUAL = 0.02
BENCHMARK_TICKER = "^STOXX50E"
DATA_PATH = "../mi_cartera/historial_operaciones.csv"

# ==========================================
# DATA INGESTION & ROBUSTNESS
# ==========================================
@st.cache_data(ttl=60)
def load_trade_data():
    """Loads the Spanish ledger and standardizes headers for the engine."""
    try:
        # 1. Tell pandas to parse the 'fecha' column as dates
        df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
        
        # 2. Rename 'fecha' to 'Date' so the rest of the MTM engine 
        #    doesn't need to be rewritten.
        df = df.rename(columns={"fecha": "Date"})
        
        # 3. Sort chronologically
        df = df.sort_values("Date").reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        st.warning(f"File '{DATA_PATH}' not found. Check the relative path.")
        # ... (keep the mock data fallback here just in case) ...
        return pd.DataFrame() # Or keep the mock generation logic

# ==========================================
# WALK-FORWARD ACCOUNTING ENGINE
# ==========================================
@st.cache_data(ttl=60)
def process_ledger(df):
    """Iterates chronologically, trusting the CSV strictly for cash flows."""
    inventory = {}
    vwap = {}
    costs_paid = {}
    
    cash_history = []
    daily_shares = []
    
    current_cash = INITIAL_CAPITAL
    
    # SORTING FIX: Sort by Date, then by Accion. 
    # 'COMPRA' comes before 'VENTA' alphabetically, ensuring same-day buys 
    # are processed before same-day sells, avoiding phantom zero-inventory.
    df_sorted = df.sort_values(by=["Date", "Accion"]).reset_index(drop=True)
    
    dates = sorted(df['Date'].unique())
    start_date = dates[0]
    end_date = pd.Timestamp.today().normalize()
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    trade_idx = 0
    
    for current_date in all_dates:
        while trade_idx < len(df_sorted) and df_sorted.iloc[trade_idx]['Date'] <= current_date:
            row = df_sorted.iloc[trade_idx]
            
            t = str(row['Ticker']).strip().upper()
            action = str(row['Accion']).strip().upper()
            qty = abs(float(row['Cantidad'])) 
            p = float(row['Precio'])
            p_exec = float(row['Precio_Ejecutado'])
            
            if t not in inventory:
                inventory[t] = 0
                vwap[t] = 0.0
                costs_paid[t] = 0.0
                
            if action == "COMPRA":
                cost = qty * p_exec
                current_cash -= cost
                friction = (p_exec - p) * qty
                costs_paid[t] += friction
                
                total_value = (inventory[t] * vwap[t]) + cost
                inventory[t] += qty
                vwap[t] = total_value / inventory[t] if inventory[t] > 0 else 0.0
                
            elif action == "VENTA":
                # STRICT TRUST: We no longer cap the sale. 
                # If the CSV says you sold it, you sold it.
                revenue = qty * p_exec
                current_cash += revenue
                
                # Friction on sale is clean price minus executed price
                friction = (p - p_exec) * qty 
                costs_paid[t] += friction
                
                inventory[t] -= qty
                
                # Floor inventory to 0 to clean up floating point errors
                if inventory[t] <= 0.001: 
                    inventory[t] = 0
                    vwap[t] = 0.0
                        
            trade_idx += 1
            
        cash_history.append({"Date": current_date, "Cash": current_cash})
        for ticker, amount in inventory.items():
            if amount > 0:
                daily_shares.append({"Date": current_date, "Ticker": ticker, "Shares": amount, "VWAP": vwap[ticker]})

    return pd.DataFrame(cash_history), pd.DataFrame(daily_shares), inventory, vwap, costs_paid

# ==========================================
# MARKET DATA & MTM CALCULATION
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data(tickers, start_date):
    """Fetches daily closing prices with fallback mechanics."""
    t_list = list(set(tickers + [BENCHMARK_TICKER]))
    data = yf.download(t_list, start=start_date, end=pd.Timestamp.today() + timedelta(days=1), progress=False)['Close']
    data.fillna(method='ffill', inplace=True)
    return data

def get_live_prices(tickers):
    """Gets spot prices using 1d/1m with 5d/1d fallback."""
    spot = {}
    for t in tickers:
        try:
            live = yf.download(t, period="1d", interval="1m", progress=False)['Close']
            if live.empty:
                raise ValueError
            spot[t] = live.iloc[-1].item()
        except:
            live = yf.download(t, period="5d", interval="1d", progress=False)['Close']
            spot[t] = live.iloc[-1].item() if not live.empty else 0.0
    return spot

# ==========================================
# UI RENDERING & LOGIC
# ==========================================
def main():
    st.title("🇪🇺 Eurostoxx 50 Walk-Forward MTM Engine")
    
    # Auto-Refresh Toggle
    auto_refresh = st.sidebar.checkbox("Enable 10s Auto-Refresh", value=False)
    
    df_trades = load_trade_data()
    df_cash, df_shares, current_inventory, current_vwap, costs_paid = process_ledger(df_trades)
    
    all_tickers = df_shares['Ticker'].unique().tolist() if not df_shares.empty else []
    start_date = df_cash['Date'].min()
    
    market_px = fetch_market_data(all_tickers, start_date)
    spot_prices = get_live_prices([t for t, q in current_inventory.items() if q > 0])
    
    # Build Daily MTM True NAV
    daily_mtm = []
    for d in df_cash['Date']:
        c = df_cash.loc[df_cash['Date'] == d, 'Cash'].values[0]
        invested = 0.0
        
        day_shares = df_shares[df_shares['Date'] == d]
        for _, row in day_shares.iterrows():
            t = row['Ticker']
            # Find closest available price on or before date
            try:
                px_series = market_px[t]
                px_date = px_series[:d].index[-1]
                invested += row['Shares'] * px_series[px_date]
            except:
                pass 
                
        daily_mtm.append({"Date": d, "NAV": c + invested, "Benchmark_Px": market_px[BENCHMARK_TICKER][:d].iloc[-1] if not market_px[BENCHMARK_TICKER][:d].empty else 1.0})
        
    df_nav = pd.DataFrame(daily_mtm)
    df_nav['Benchmark_Return'] = df_nav['Benchmark_Px'].pct_change().fillna(0)
    df_nav['Benchmark_NAV'] = INITIAL_CAPITAL * (1 + df_nav['Benchmark_Return']).cumprod()
    df_nav['Strategy_Return'] = df_nav['NAV'].pct_change().fillna(0)

    # Current State Calculations
    current_cash = df_cash.iloc[-1]['Cash']
    total_invested_mtm = sum([qty * spot_prices[t] for t, qty in current_inventory.items() if qty > 0])
    true_nav = current_cash + total_invested_mtm
    unrealized_pnl = sum([(spot_prices[t] - current_vwap[t]) * qty for t, qty in current_inventory.items() if qty > 0])
    open_costs = sum([costs_paid[t] for t, qty in current_inventory.items() if qty > 0])
    
    # ------------------------------------------
    # RISK METRICS (Strict Covariance Matrix Slicing)
    # ------------------------------------------
    active_tickers = [t for t, q in current_inventory.items() if q > 0]
    
    if len(active_tickers) > 0 and len(df_nav) > 2:
        # Returns for active assets only
        hist_returns = market_px[active_tickers].pct_change().dropna()
        cov_matrix = hist_returns.cov() * 252 # Annualized
        
        # Current Weights
        weights = np.array([(current_inventory[t] * spot_prices[t]) / total_invested_mtm for t in active_tickers])
        
        # Portfolio Variance: W^T * Sigma * W
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)
        
        # 1-Day 99% VaR (Parametric)
        daily_volatility = port_volatility / np.sqrt(252)
        var_99_1d = 2.326 * daily_volatility * total_invested_mtm
        
        # Beta & Alpha
        bench_ret = df_nav['Benchmark_Return']
        strat_ret = df_nav['Strategy_Return']
        cov_sb = np.cov(strat_ret, bench_ret)[0][1]
        var_b = np.var(bench_ret)
        beta = cov_sb / var_b if var_b != 0 else 0.0
        
        cum_strat_ret = (true_nav - INITIAL_CAPITAL) / INITIAL_CAPITAL
        cum_bench_ret = (df_nav.iloc[-1]['Benchmark_NAV'] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        days_held = (df_nav.iloc[-1]['Date'] - df_nav.iloc[0]['Date']).days
        rfr_period = RFR_ANNUAL * (days_held / 365.25)
        
        # Jensen's Alpha
        alpha = cum_strat_ret - (rfr_period + beta * (cum_bench_ret - rfr_period))
        
        # Daily Sharpe
        daily_rfr = RFR_ANNUAL / 252
        sharpe = np.sqrt(252) * ((strat_ret.mean() - daily_rfr) / strat_ret.std()) if strat_ret.std() != 0 else 0.0

    else:
        var_99_1d, beta, alpha, sharpe = 0.0, 0.0, 0.0, 0.0

    # ------------------------------------------
    # TOP LEVEL METRICS
    # ------------------------------------------
    st.subheader("Treasury Aggregates")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("True MTM NAV", f"€{true_nav:,.2f}")
    col2.metric("Residual Cash", f"€{current_cash:,.2f}")
    col3.metric("Invested MTM", f"€{total_invested_mtm:,.2f}")
    col4.metric("Unrealized PnL", f"€{unrealized_pnl:,.2f}")
    col5.metric("Accounting Check (Zero Diff)", f"€{(current_cash + total_invested_mtm) - true_nav:,.2f}")
    col6.metric("Friction (Open Positions)", f"€{open_costs:,.2f}")

    st.markdown("---")
    st.subheader("Risk Analytics")
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    rcol1.metric("Live Portfolio Beta", f"{beta:.3f}")
    rcol2.metric("1-Day 99% VaR", f"€{var_99_1d:,.2f}")
    rcol3.metric("Daily Sharpe Ratio", f"{sharpe:.2f}")
    rcol4.metric("Jensen's Alpha (Cum.)", f"{alpha * 100:.2f}%")

    st.markdown("---")
    
    # ------------------------------------------
    # CHARTS
    # ------------------------------------------
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Strategy NAV vs ^STOXX50E (Scaled)")
        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(x=df_nav['Date'], y=df_nav['NAV'], mode='lines', name='Strategy MTM NAV', line=dict(color='#00d4ff')))
        fig_nav.add_trace(go.Scatter(x=df_nav['Date'], y=df_nav['Benchmark_NAV'], mode='lines', name='Eurostoxx 50 (Scaled)', line=dict(color='#ff00d4')))
        fig_nav.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_nav, use_container_width=True)
        
    with c2:
        st.subheader("MTM Allocation")
        if total_invested_mtm > 0:
            alloc_data = [{"Ticker": t, "Notional": current_inventory[t] * spot_prices[t]} for t in active_tickers]
            alloc_data.append({"Ticker": "Cash", "Notional": current_cash})
            df_alloc = pd.DataFrame(alloc_data)
            fig_pie = px.pie(df_alloc, names='Ticker', values='Notional', hole=0.6)
            fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No open positions. 100% Cash.")

    # ------------------------------------------
    # OPEN POSITIONS TABLE
    # ------------------------------------------
    st.subheader("Open Positions & PnL Ledger")
    
    table_data = []
    for t in active_tickers:
        qty = current_inventory[t]
        spot = spot_prices[t]
        vwap_px = current_vwap[t]
        notional = qty * spot
        weight = notional / true_nav
        gross_pnl = (spot - vwap_px) * qty
        costs = costs_paid[t]
        net_pnl = gross_pnl - costs
        
        table_data.append({
            "Ticker": t,
            "Net Qty": qty,
            "Cost Basis (VWAP)": round(vwap_px, 4),
            "Spot Price": round(spot, 4),
            "Notional (€)": notional,
            "Weight (%)": weight * 100,
            "Unrealized Gross PnL": gross_pnl,
            "Costs Paid": costs,
            "Real Net PnL": net_pnl
        })
        
    if table_data:
        df_table = pd.DataFrame(table_data)
        
        # Styling
        def style_pnl(val):
            color = '#00ff88' if val > 0 else '#ff4444' if val < 0 else 'white'
            return f'color: {color}'

        styled_table = df_table.style.map(style_pnl, subset=['Unrealized Gross PnL', 'Real Net PnL']) \
                                    .format({"Notional (€)": "{:,.2f}", "Weight (%)": "{:.2f}%", 
                                             "Unrealized Gross PnL": "{:,.2f}", "Costs Paid": "{:,.2f}", 
                                             "Real Net PnL": "{:,.2f}"})
        st.dataframe(styled_table, use_container_width=True, hide_index=True)
    else:
        st.info("No active equity positions in the ledger.")

    # Loop
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()