# simulatore_etf_picker_monthly.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, Tuple

st.set_page_config(page_title="ETF Picker - Monthly Simulation", layout="wide")
st.title("ETF Picker 2.0 — Acquisti mensili, commissioni Fineco, inflazione, tasse")

# ---------------------------
# Helpers & Cache
# ---------------------------
@st.cache_data(ttl=6*3600)
def fetch_price_history_cached(ticker: str, period: str = "max") -> pd.DataFrame:
    """Fetch adjusted close price history via yfinance and cache results."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"Nessun storico trovato per {ticker}")
    # prefer 'Close' which with auto_adjust=True is adjusted close
    hist = hist.rename(columns={c: c for c in hist.columns})
    hist = hist[["Close"]].dropna()
    hist.index = pd.to_datetime(hist.index)
    return hist

def compute_cagr_vol_from_prices(price_series: pd.Series) -> Tuple[float, float]:
    """Return CAGR and annualized volatility from price series (daily)."""
    # daily returns
    ret = price_series.pct_change().dropna()
    if ret.empty:
        return 0.0, 0.0
    trading_days = 252.0
    avg_daily = ret.mean()
    std_daily = ret.std()
    ann_return_approx = (1 + avg_daily) ** trading_days - 1
    ann_vol = std_daily * np.sqrt(trading_days)
    # CAGR better estimate:
    years = (price_series.index[-1] - price_series.index[0]).days / 365.25
    if years > 0:
        cagr = (price_series.iloc[-1] / price_series.iloc[0]) ** (1/years) - 1
    else:
        cagr = ann_return_approx
    return float(cagr), float(ann_vol)

def commission_for_trade(eur_amount: float, in_usd: bool, fineco_commission_eur: float, fx_spread: float) -> float:
    """Simplified commission model: fixed commission if <2500, plus FX spread if USD."""
    cost = 0.0
    if eur_amount <= 0:
        return 0.0
    if eur_amount < 2500.0:
        cost += fineco_commission_eur
    if in_usd:
        cost += eur_amount * fx_spread
    return cost

# ---------------------------
# Preloaded ETF examples (you can change or remove)
# ---------------------------
PRELOADED = {
    # ticker : {name, ter}
    "VWCE.DE": {"name": "Vanguard FTSE All-World UCITS ETF (Acc) - VWCE", "ter": 0.0019},
    # add more preloaded if you want
}

# ---------------------------
# Sidebar: ETF input & params
# ---------------------------
st.sidebar.header("1) Aggiungi / Carica ETF")
st.sidebar.write("Inserisci ticker (es. VWCE.DE) o ISIN (es. IE00BK5BQT80). Preferibile ticker.")
user_input = st.sidebar.text_input("Ticker o ISIN", value="VWCE.DE")
if "loaded_etfs" not in st.session_state:
    st.session_state.loaded_etfs = {}
if st.sidebar.button("Carica ETF"):
    q = user_input.strip().upper()
    ticker = None
    # if input looks like ISIN (12 chars starting with letters) try to guess ticker by simple fallback:
    if len(q) == 12 and q[:2].isalpha():
        # try using yfinance search fallback - just try q as ticker first
        # Many ISINs won't map directly; prefer ticker insert by user
        st.sidebar.info("Se hai inserito ISIN prova ad inserire il ticker (es. VWCE.DE). Sto tentando comunque la ricerca.")
        ticker = q  # try, will likely fail and user can retry with ticker
    else:
        ticker = q

    try:
        hist = fetch_price_history_cached(ticker)
        cagr, vol = compute_cagr_vol_from_prices(hist["Close"])
        ter_default = PRELOADED.get(ticker, {}).get("ter", 0.005)  # fallback TER 0.5%
        st.session_state.loaded_etfs[ticker] = {
            "ticker": ticker,
            "name": PRELOADED.get(ticker, {}).get("name", ticker),
            "price": hist["Close"],
            "cagr": cagr,
            "vol": vol,
            "ter": ter_default
        }
        st.sidebar.success(f"Caricato {ticker} — CAGR {cagr:.2%}, Vol {vol:.2%}, TER default {ter_default*100:.2f}%")
    except Exception as e:
        st.sidebar.error(f"Errore caricamento {ticker}: {e}")

st.sidebar.markdown("---")
st.sidebar.header("ETF caricati")
if len(st.session_state.loaded_etfs) == 0:
    st.sidebar.info("Nessun ETF caricato. Carica almeno uno per costruire il portafoglio.")
else:
    for t, d in st.session_state.loaded_etfs.items():
        st.sidebar.write(f"• {t} — {d['name']}")
        st.sidebar.write(f"   CAGR {d['cagr']:.2%} • Vol {d['vol']:.2%} • TER {d['ter']*100:.2f}%")
        if st.sidebar.button(f"Rimuovi {t}", key=f"rm_{t}"):
            st.session_state.loaded_etfs.pop(t)
            st.rerun()

# ---------------------------
# Sidebar: investment params & Fineco costs
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("2) Parametri investimento")
initial_capital = st.sidebar.number_input("Capitale iniziale (€)", value=1000.0, step=100.0)
monthly_input = st.sidebar.number_input("Versamento mensile (€)", value=50.0, step=10.0)
horizon = st.sidebar.slider("Orizzonte (anni)", 1, 40, 20)
scenario = st.sidebar.selectbox("Scenario", ["Neutro", "Pessimistico", "Ottimistico"], index=0)
tax_rate = st.sidebar.number_input("Aliquota capital gain (%)", value=26.0, step=0.5) / 100.0
inflation = st.sidebar.number_input("Inflazione annua prevista (%)", value=2.0, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("3) Costi Fineco & FX")
fineco_commission_eur = st.sidebar.number_input("Commissione Fineco per ordine (€)", value=2.95, step=0.01)
fx_spread = st.sidebar.number_input("Spread cambio EUR→USD (%)", value=0.25, step=0.01) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("4) Simulazione e target")
execution_mode = st.sidebar.selectbox("Esecuzione acquisti", ["Monthly (accurato)", "Annual (veloce)"])
target_active = st.sidebar.checkbox("Imposta target finanziario", value=True)
target_value = st.sidebar.number_input("Target (€)", value=100000.0, step=1000.0)
st.sidebar.markdown("---")

# ---------------------------
# Build portfolio allocations UI
# ---------------------------
st.header("Configura portafoglio")
if len(st.session_state.loaded_etfs) == 0:
    st.info("Carica almeno un ETF dalla sidebar per procedere.")
    st.stop()

etf_list = list(st.session_state.loaded_etfs.keys())
st.subheader("Allocazioni (devono sommare a 100%)")
alloc_inputs = {}
col_allocs = st.columns(len(etf_list))
for i, t in enumerate(etf_list):
    with col_allocs[i]:
        default = int(100 / len(etf_list))
        alloc_inputs[t] = st.number_input(f"% {t}", min_value=0, max_value=100, value=default, step=1, key=f"alloc_{t}")

alloc_array = np.array([alloc_inputs[t] for t in etf_list], dtype=float)
if alloc_array.sum() == 0:
    st.error("Imposta almeno una allocazione maggiore di 0%")
    st.stop()
alloc_norm = alloc_array / alloc_array.sum()

# Allow TER override per ETF
st.subheader("TER per ETF (override se necessario)")
for t in etf_list:
    ter_val = st.number_input(f"TER {t} (%)", value=float(st.session_state.loaded_etfs[t]["ter"]*100), step=0.01, key=f"ter_{t}")
    st.session_state.loaded_etfs[t]["ter"] = ter_val / 100.0

# Scenario multipliers
mult_rend = {"Pessimistico": 0.7, "Neutro": 1.0, "Ottimistico": 1.25}.get(scenario, 1.0)

# ---------------------------
# Prepare arrays for simulation
# ---------------------------
cagrs = np.array([st.session_state.loaded_etfs[t]["cagr"] for t in etf_list]) * mult_rend
vols = np.array([st.session_state.loaded_etfs[t]["vol"] for t in etf_list])
ters = np.array([st.session_state.loaded_etfs[t]["ter"] for t in etf_list])

# ---------------------------
# Simulation functions
# ---------------------------
def monthly_simulation(initial_capital: float,
                       monthly_input: float,
                       horizon_years: int,
                       alloc_norm: np.ndarray,
                       cagrs: np.ndarray,
                       ters: np.ndarray,
                       fineco_commission_eur: float,
                       fx_spread: float,
                       etf_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate month-by-month purchases and NAV growth."""
    months = horizon_years * 12
    nav = np.zeros(months)
    invested = np.zeros(months)
    nav_val = float(initial_capital)
    invested_cum = float(initial_capital)
    # convert annual cagrs to monthly rates per ETF (approx)
    monthly_rates = (1 + cagrs) ** (1 / 12) - 1
    # approximate TER monthly drag (assume TER reduces returns multiplicatively)
    monthly_ter_drag = (1 - ters) ** (1 / 12) - 1  # negative number approx
    for m in range(months):
        # monthly buy at beginning of month
        total_commission = 0.0
        for j, t in enumerate(etf_list):
            # naive USD detection from ticker string (improvable)
            in_usd = ("USD" in t.upper()) or (".US" in t.upper()) or ("NYSE" in t.upper())
            tranche = monthly_input * alloc_norm[j]
            total_commission += commission_for_trade(tranche, in_usd, fineco_commission_eur, fx_spread)
        net_buy = monthly_input - total_commission
        if net_buy < 0:
            net_buy = 0.0
        nav_val += net_buy
        invested_cum += monthly_input
        # apply growth: combine weighted monthly rates
        # weighted rate = dot(alloc, monthly_rates)
        weighted_monthly_return = float(np.dot(alloc_norm, monthly_rates))
        # apply ter drag approx (weighted)
        weighted_ter_drag = float(np.dot(alloc_norm, monthly_ter_drag))
        nav_val = nav_val * (1.0 + weighted_monthly_return + weighted_ter_drag)
        nav[m] = nav_val
        invested[m] = invested_cum
    return nav, invested

def annual_simulation(initial_capital: float,
                      annual_input: float,
                      horizon_years: int,
                      alloc_norm: np.ndarray,
                      cagrs: np.ndarray,
                      ters: np.ndarray,
                      fineco_commission_eur: float,
                      fx_spread: float,
                      etf_list: list) -> Tuple[np.ndarray, np.ndarray]:
    years = horizon_years
    nav = np.zeros(years)
    invested = np.zeros(years)
    nav_val = float(initial_capital)
    invested_cum = float(initial_capital)
    annual_rates = cagrs
    annual_ter_drag = (1 - ters) - 1  # approx negative
    for y in range(years):
        # buy at beginning (annual aggregated contribution)
        total_commission = 0.0
        for j, t in enumerate(etf_list):
            in_usd = ("USD" in t.upper()) or (".US" in t.upper()) or ("NYSE" in t.upper())
            tranche = annual_input * alloc_norm[j]
            total_commission += commission_for_trade(tranche, in_usd, fineco_commission_eur, fx_spread)
        net_buy = annual_input - total_commission
        if net_buy < 0:
            net_buy = 0.0
        nav_val += net_buy
        invested_cum += annual_input
        # apply growth
        weighted_annual_return = float(np.dot(alloc_norm, annual_rates))
        weighted_ter = float(np.dot(alloc_norm, ters))
        nav_val = nav_val * (1.0 + weighted_annual_return) * (1.0 - weighted_ter)
        nav[y] = nav_val
        invested[y] = invested_cum
    return nav, invested

# ---------------------------
# Run chosen simulation
# ---------------------------
if execution_mode.startswith("Monthly"):
    nav_monthly, invested_monthly = monthly_simulation(
        initial_capital=initial_capital,
        monthly_input=monthly_input,
        horizon_years=horizon,
        alloc_norm=alloc_norm,
        cagrs=cagrs,
        ters=ters,
        fineco_commission_eur=fineco_commission_eur,
        fx_spread=fx_spread,
        etf_list=etf_list
    )
    # compress monthly arrays to yearly for summary table (end of each year)
    months = horizon * 12
    years_idx = np.arange(1, horizon + 1)
    nav_yearly = np.array([nav_monthly[(y*12)-1] for y in range(1, horizon+1)])
    invested_yearly = np.array([invested_monthly[(y*12)-1] for y in range(1, horizon+1)])
    timeline_years = years_idx
else:
    # approximate annual contributions from monthly_input
    annual_input = monthly_input * 12.0
    nav_yearly, invested_yearly = annual_simulation(
        initial_capital=initial_capital,
        annual_input=annual_input,
        horizon_years=horizon,
        alloc_norm=alloc_norm,
        cagrs=cagrs,
        ters=ters,
        fineco_commission_eur=fineco_commission_eur,
        fx_spread=fx_spread,
        etf_list=etf_list
    )
    timeline_years = np.arange(1, horizon+1)

# Final metrics (nominal)
final_nominal = float(nav_yearly[-1])
total_invested = float(invested_yearly[-1])
gain_nominal = final_nominal - total_invested
tax_amount = max(0.0, gain_nominal) * tax_rate
final_after_tax = final_nominal - tax_amount

# Real (inflation-adjusted)
real_final = final_nominal / ((1 + inflation) ** horizon)
real_after_tax = final_after_tax / ((1 + inflation) ** horizon)

# CAGR (on invested)
def compute_cagr(start_value, end_value, years): 
    if start_value <= 0 or years <= 0:
        return 0.0
    return (end_value / start_value) ** (1.0 / years) - 1.0

cagr_nominal = compute_cagr(total_invested, final_nominal, horizon)
cagr_real = compute_cagr(total_invested, real_after_tax, horizon)

# Output KPIs
st.markdown("## Risultati principali")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Valore finale (nominale)", f"€ {final_nominal:,.0f}")
col2.metric("Totale investito", f"€ {total_invested:,.0f}")
col3.metric("Profitto (nominale)", f"€ {gain_nominal:,.0f}")
col4.metric("Valore netto dopo tasse", f"€ {final_after_tax:,.0f}")
col5, col6 = st.columns(2)
col5.metric("Valore finale (reale, dopo inflazione)", f"€ {real_after_tax:,.0f}")
col6.metric("CAGR reale annuo (su investito)", f"{cagr_real*100:.2f}%")

# Target check
if target_active:
    reached = final_after_tax >= target_value
    st.info(f"Target = € {target_value:,.0f} → Deterministic: {'Raggiunto' if reached else 'Non raggiunto'}")

# ---------------------------
# Build dataframe year-by-year
# ---------------------------
df = pd.DataFrame({
    "Anno": timeline_years,
    "Investito cumulato (€)": invested_yearly,
    "Valore nominale (€)": nav_yearly
})
df["Profitto (€)"] = df["Valore nominale (€)"] - df["Investito cumulato (€)"]
df["Valore netto dopo tasse (€)"] = df["Valore nominale (€)"] - (np.maximum(0, df["Profitto (€)"]) * tax_rate)
df["Valore reale (€)"] = df["Valore nominale (€)"] / ((1.0 + inflation) ** df["Anno"].values)

# ---------------------------
# Plot interactive with Plotly
# ---------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Anno"], y=df["Investito cumulato (€)"], mode="lines+markers",
                         name="Investito", line=dict(dash="dash", color="#00C9A7")))
fig.add_trace(go.Scatter(x=df["Anno"], y=df["Valore nominale (€)"], mode="lines+markers",
                         name="Valore lordo", line=dict(color="#FF7F50", width=4)))
fig.add_trace(go.Scatter(x=df["Anno"], y=df["Valore netto dopo tasse (€)"], mode="lines+markers",
                         name="Valore netto dopo tasse", line=dict(color="#4B8BFF", width=3)))
if target_active:
    fig.add_hline(y=target_value, line_dash="dot", annotation_text="Target", annotation_position="top right")

fig.update_layout(title="Investito vs Valore (anno per anno)", xaxis_title="Anno", yaxis_title="€", template="plotly_dark", height=560)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Tabella anno per anno")
st.dataframe(df.style.format("{:,.0f}"))

# ---------------------------
# CSV save & compare
# ---------------------------
col_a, col_b = st.columns(2)
with col_a:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica simulazione (CSV)", data=csv, file_name=f"simulazione_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
with col_b:
    uploaded = st.file_uploader("Carica una simulazione (CSV) per confronto", type=["csv"])
    if uploaded is not None:
        other = pd.read_csv(uploaded)
        if "Anno" in other.columns and ("Valore nominale (€)" in other.columns or "Valore" in other.columns):
            # attempt to find a value column
            if "Valore nominale (€)" in other.columns:
                other_vals = other["Valore nominale (€)"]
            else:
                # try generic second column
                other_vals = other.iloc[:, 1]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["Anno"], y=df["Valore nominale (€)"], name="Simulazione attuale"))
            fig2.add_trace(go.Scatter(x=other["Anno"], y=other_vals, name="Simulazione caricata"))
            fig2.update_layout(template="plotly_dark", title="Confronto simulazioni", height=520)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("CSV non compatibile. Deve contenere 'Anno' e 'Valore nominale (€)'")

st.caption("Script locale: per performance migliori, evita richieste yfinance ripetute e usa cache. Per acquisti mensili molti calcoli avverranno localmente e possono richiedere CPU in caso di orizzonti molto lunghi.")

