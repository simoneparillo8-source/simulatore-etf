import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Tuple

st.set_page_config(page_title="ETF Picker — Slim", layout="wide")
st.title("ETF Picker — Slim & Realistico")

# --------------------
# Helpers & cache
# --------------------
@st.cache_data(ttl=6*3600)
def fetch_prices(ticker: str, period: str = "max") -> pd.Series:
    t = yf.Ticker(ticker)
    h = t.history(period=period, auto_adjust=True)
    if h is None or h.empty:
        raise ValueError(f"No history for {ticker}")
    s = h["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    return s

def cagr_vol(series: pd.Series) -> Tuple[float, float]:
    r = series.pct_change().dropna()
    if r.empty:
        return 0.0, 0.0
    td = 252.0
    ann_ret_approx = (1 + r.mean()) ** td - 1
    ann_vol = r.std() * np.sqrt(td)
    years = (series.index[-1] - series.index[0]).days / 365.25
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) if years > 0 else ann_ret_approx
    return float(cagr), float(ann_vol)

def commission(eur_amount: float, in_usd: bool, fineco_fee: float, fx_spread: float) -> float:
    if eur_amount <= 0:
        return 0.0
    c = fineco_fee if eur_amount < 2500.0 else 0.0
    if in_usd:
        c += eur_amount * fx_spread
    return c

def compute_cagr(start, end, years):
    if start <= 0 or years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


# --------------------
# Simulation functions (CORRETTE)
# --------------------
def simulate_monthly(initial, monthly, years, weights, cagrs, ters, fineco_fee, fx_spread, tickers):
    months = years * 12
    nav = np.zeros(months)
    invested = np.zeros(months)
    nav_val = float(initial)
    invested_cum = float(initial)
    monthly_rates = (1 + cagrs) ** (1 / 12) - 1
    monthly_ter = (1 - ters) ** (1 / 12) - 1  # TER come perdita
    usd_flags = np.array([("USD" in t.upper()) or (".US" in t.upper()) or ("USD" in st.session_state.etfs[t]["name"].upper()) for t in tickers])

    for m in range(months):
        total_comm = 0.0
        for j in range(len(tickers)):
            tranche = monthly * weights[j]
            total_comm += commission(tranche, bool(usd_flags[j]), fineco_fee, fx_spread)
        net_buy = max(0.0, monthly - total_comm)
        nav_val += net_buy
        invested_cum += monthly
        weighted_ret = float(np.dot(weights, monthly_rates))
        weighted_ter = float(np.dot(weights, monthly_ter))
        # 🔧 TER ora sottratto (non aggiunto)
        nav_val = nav_val * (1.0 + weighted_ret - abs(weighted_ter))
        nav[m] = nav_val
        invested[m] = invested_cum

    nav_year = np.array([nav[(y+1)*12 - 1] for y in range(years)])
    invested_year = np.array([invested[(y+1)*12 - 1] for y in range(years)])
    return nav_year, invested_year

def simulate_annual(initial, monthly, years, weights, cagrs, ters, fineco_fee, fx_spread, tickers):
    nav = np.zeros(years)
    invested = np.zeros(years)
    nav_val = float(initial)
    invested_cum = float(initial)
    usd_flags = np.array([("USD" in t.upper()) or (".US" in t.upper()) or ("USD" in st.session_state.etfs[t]["name"].upper()) for t in tickers])
    annual_input = monthly * 12.0

    for y in range(years):
        total_comm = 0.0
        for j in range(len(tickers)):
            tranche = annual_input * weights[j]
            total_comm += commission(tranche, bool(usd_flags[j]), fineco_fee, fx_spread)
        net_buy = max(0.0, annual_input - total_comm)
        nav_val += net_buy
        invested_cum += annual_input
        weighted_ret = float(np.dot(weights, cagrs))
        weighted_ter = float(np.dot(weights, ters))
        # 🔧 TER correttamente sottratto
        nav_val = nav_val * ((1.0 + weighted_ret) * (1.0 - abs(weighted_ter)))
        nav[y] = nav_val
        invested[y] = invested_cum

    return nav, invested


# --------------------
# UI sidebar: load ETF
# --------------------
st.sidebar.header("1) Carica ETF (ticker preferibile)")
if "etfs" not in st.session_state:
    st.session_state.etfs = {}

ticker_in = st.sidebar.text_input("Ticker (es. VWCE.DE, IE00BK5BQT80 -> preferisci ticker)", value="VWCE.DE")
if st.sidebar.button("Carica"):
    q = ticker_in.strip().upper()
    try:
        prices = fetch_prices(q)
        c, v = cagr_vol(prices)
        ter_default = 0.0019 if "VWCE" in q else 0.005
        st.session_state.etfs[q] = {
            "ticker": q,
            "name": q,
            "prices": prices,
            "cagr": c,
            "vol": v,
            "ter": ter_default
        }
        st.sidebar.success(f"Caricato {q} — CAGR {c:.2%}, vol {v:.2%}")
    except Exception as e:
        st.sidebar.error(f"Errore: {e}")

if len(st.session_state.etfs) > 0:
    st.sidebar.markdown("**ETF in memoria**")
    for t, info in list(st.session_state.etfs.items()):
        st.sidebar.write(f"- {t} — CAGR {info['cagr']:.2%} — TER {info['ter']*100:.2f}%")
        if st.sidebar.button(f"Rimuovi {t}", key=f"rm_{t}"):
            st.session_state.etfs.pop(t)
            st.experimental_rerun()
else:
    st.sidebar.info("Carica almeno 1 ETF")

# --------------------
# Sidebar: params
# --------------------
st.sidebar.header("2) Parametri investimento")
initial = st.sidebar.number_input("Capitale iniziale (€)", value=1000.0, step=100.0)
monthly = st.sidebar.number_input("Versamento mensile (€)", value=100.0, step=10.0)
horizon = st.sidebar.slider("Orizzonte (anni)", 1, 40, 30)
scenario = st.sidebar.selectbox("Scenario", ["Neutro", "Pessimistico", "Ottimistico"])
tax = st.sidebar.number_input("Aliquota capital gain (%)", value=26.0, step=0.5) / 100.0
infl = st.sidebar.number_input("Inflazione annua (%)", value=2.0, step=0.1) / 100.0

st.sidebar.header("3) Costi Fineco")
fineco_fee = st.sidebar.number_input("Commissione per ordine (€)", value=2.95, step=0.01)
fx_spread = st.sidebar.number_input("Spread cambio EUR→USD (%)", value=0.25, step=0.01) / 100.0

st.sidebar.header("4) Simulazione")
mode = st.sidebar.selectbox("Esecuzione acquisti", ["Monthly (accurato)", "Annual (veloce)"])
target_active = st.sidebar.checkbox("Imposta target", value=True)
target = st.sidebar.number_input("Target (€)", value=100000.0, step=1000.0)

# --------------------
# Portfolio allocations
# --------------------
st.header("Configura portafoglio")
if len(st.session_state.etfs) == 0:
    st.info("Carica almeno un ETF nella sidebar.")
    st.stop()

tickers = list(st.session_state.etfs.keys())
cols = st.columns(len(tickers))
alloc = {}
for i, t in enumerate(tickers):
    with cols[i]:
        default = int(100 / len(tickers))
        alloc[t] = st.slider(f"% {t}", 0, 100, default, key=f"alloc_{t}")
alloc_arr = np.array([alloc[t] for t in tickers], dtype=float)
if alloc_arr.sum() == 0:
    st.error("Imposta almeno una percentuale > 0")
    st.stop()
weights = alloc_arr / alloc_arr.sum()

# TER override
for t in tickers:
    ter_val = st.number_input(f"TER {t} (%)", value=st.session_state.etfs[t]["ter"]*100, key=f"ter_{t}") / 100.0
    st.session_state.etfs[t]["ter"] = ter_val

# Scenario adjustment
mult = {"Pessimistico": 0.7, "Neutro": 1.0, "Ottimistico": 1.25}[scenario]
cagrs = np.array([st.session_state.etfs[t]["cagr"] for t in tickers]) * mult
ters = np.array([st.session_state.etfs[t]["ter"] for t in tickers])

# --------------------
# Simulation execution
# --------------------
if mode.startswith("Monthly"):
    nav_y, inv_y = simulate_monthly(initial, monthly, horizon, weights, cagrs, ters, fineco_fee, fx_spread, tickers)
else:
    nav_y, inv_y = simulate_annual(initial, monthly, horizon, weights, cagrs, ters, fineco_fee, fx_spread, tickers)

final_nom = float(nav_y[-1])
invested_total = float(inv_y[-1])
gain_nom = final_nom - invested_total
tax_amt = max(0.0, gain_nom) * tax
final_after = final_nom - tax_amt
real_after = final_after / ((1 + infl) ** horizon)
cagr_nom = compute_cagr(invested_total, final_nom, horizon)
cagr_real = compute_cagr(invested_total, real_after, horizon)

# --------------------
# Output & plots
# --------------------
st.subheader("Risultati")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Valore finale (nom)", f"€ {final_nom:,.0f}")
c2.metric("Investito", f"€ {invested_total:,.0f}")
c3.metric("Profitto", f"€ {gain_nom:,.0f}")
c4.metric("Netto dopo tasse", f"€ {final_after:,.0f}")
st.write(f"CAGR reale annuo (su investito): {cagr_real*100:.2f}%")

years_idx = np.arange(1, horizon+1)
df = pd.DataFrame({
    "Anno": years_idx,
    "Investito (€)": inv_y,
    "Valore nominale (€)": nav_y
})
df["Profitto (€)"] = df["Valore nominale (€)"] - df["Investito (€)"]
df["Netto dopo tasse (€)"] = df["Valore nominale (€)"] - (np.maximum(0, df["Profitto (€)"]) * tax)
df["Valore reale (€)"] = df["Valore nominale (€)"] / ((1 + infl) ** df["Anno"].values)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Anno"], y=df["Investito (€)"], name="Investito", line=dict(color="#00C9A7", dash="dash")))
fig.add_trace(go.Scatter(x=df["Anno"], y=df["Valore nominale (€)"], name="Valore lordo", line=dict(color="#FF7F50", width=3)))
fig.add_trace(go.Scatter(x=df["Anno"], y=df["Netto dopo tasse (€)"], name="Netto dopo tasse", line=dict(color="#4B8BFF", width=2)))
if target_active:
    fig.add_hline(y=target, line=dict(dash="dot", color="white"), annotation_text="Target", annotation_position="top right")
fig.update_layout(template="plotly_dark", height=520, xaxis_title="Anno", yaxis_title="€")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Tabella anno-per-anno")
st.dataframe(df.style.format("{:,.0f}"))

c_left, c_right = st.columns(2)
with c_left:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV", csv, file_name=f"sim_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
with c_right:
    up = st.file_uploader("Carica CSV per confronto", type=["csv"])
    if up is not None:
        other = pd.read_csv(up)
        if "Anno" in other.columns and ("Valore nominale (€)" in other.columns or other.shape[1] >= 2):
            ycol = "Valore nominale (€)" if "Valore nominale (€)" in other.columns else other.columns[1]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["Anno"], y=df["Valore nominale (€)"], name="Attuale"))
            fig2.add_trace(go.Scatter(x=other["Anno"], y=other[ycol], name="Caricata"))
            fig2.update_layout(template="plotly_dark", height=480)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("CSV non compatibile. Serve 'Anno' e colonna valore.")

st.caption("Versione corretta: TER sottratto dal rendimento, valori realistici per investimenti a lungo termine.")
