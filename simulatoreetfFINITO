# simulatore_etf_avanzato.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------
# Config Streamlit
# -----------------------
st.set_page_config(page_title="Simulatore ETF Avanzato", layout="wide", page_icon="💹")
st.title("💹 Simulatore ETF Avanzato — tasse, inflazione, cambio e target")
st.markdown("Versione con simulazione deterministica + Monte Carlo (opzionale).")

# -----------------------
# Sidebar - Input utente
# -----------------------
st.sidebar.header("Parametri generali")
horizon = st.sidebar.slider("Orizzonte (anni)", 1, 40, 20)
scenario = st.sidebar.radio("Scenario di mercato", ["Pessimistico", "Neutro", "Ottimistico"], index=1)

st.sidebar.markdown("### Tasse & Inflazione")
tax_rate = st.sidebar.number_input("Aliquota capital gain (%)", min_value=0.0, max_value=100.0, value=26.0, step=0.5) / 100.0
inflation = st.sidebar.number_input("Inflazione annua attesa (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100.0

st.sidebar.markdown("### Cambio EUR/USD (per ETF non hedged)")
fx_initial = st.sidebar.number_input("Prezzo iniziale EUR/USD (1 EUR = ? USD)", value=1.0, step=0.01, format="%.3f")
fx_drift = st.sidebar.number_input("Drift annuale atteso USD in EUR (%) (es: 0.02 = USD +2% p.a.)", value=0.00, step=0.01)  # expressed in fraction
fx_vol = st.sidebar.number_input("Volatilità FX annua (%)", value=8.0, step=0.5) / 100.0

st.sidebar.markdown("### Monte Carlo (opzionale)")
mc_runs = st.sidebar.slider("Numero simulazioni Monte Carlo (0 = disattiva)", 0, 5000, 0, step=50)
mc_seed = st.sidebar.number_input("Seed (0 per random)", value=0, step=1)

st.sidebar.markdown("### Target")
target_active = st.sidebar.checkbox("Imposta target finanziario", value=True)
target_value = st.sidebar.number_input("Target (€)", min_value=0.0, value=50000.0, step=100.0)

st.sidebar.markdown("---")

# -----------------------
# ETF realistici (fixed)
# -----------------------
st.sidebar.header("Allocazioni ETF (modifica solo i pesi)")
etf_data = {
    "Vanguard FTSE All-World (VWCE) [acc]": {"mu": 0.063, "vol": 0.135, "hedged": False},  # global
    "S&P 500 (USD) [non-hedged]": {"mu": 0.070, "vol": 0.140, "hedged": False},
    "Europa / Developed": {"mu": 0.055, "vol": 0.110, "hedged": True},
    "IA / Tech (tematico)": {"mu": 0.085, "vol": 0.180, "hedged": False},
}

# default allocations
defaults = [35, 30, 20, 15]
etf_names = list(etf_data.keys())
alloc_inputs = []
for i, name in enumerate(etf_names):
    alloc_inputs.append(st.sidebar.slider(name + " (%)", 0, 100, defaults[i], step=1))

alloc_array = np.array(alloc_inputs, dtype=float)
if alloc_array.sum() == 0:
    st.sidebar.error("Imposta almeno una percentuale > 0")
    st.stop()
alloc_norm = alloc_array / alloc_array.sum()

# -----------------------
# Adjustments per scenario
# -----------------------
mult_rend = {"Pessimistico": 0.7, "Neutro": 1.0, "Ottimistico": 1.25}[scenario]
mult_vol = {"Pessimistico": 1.25, "Neutro": 1.0, "Ottimistico": 0.85}[scenario]

# build arrays
mus = np.array([etf_data[n]["mu"] for n in etf_names]) * mult_rend
vols = np.array([etf_data[n]["vol"] for n in etf_names]) * mult_vol
hedged_flags = np.array([etf_data[n]["hedged"] for n in etf_names])

# -----------------------
# Cashflow schedule (monthly contributions converted yearly)
# Based on your earlier plan: 0 in years 1-5, 50€/month years 6-10, 100€/month years 11-... etc.
# Allow longer contributions: after year 20 we use 150€/month as earlier versions used.
# -----------------------
monthly = np.zeros(horizon, dtype=float)
for y in range(1, horizon+1):
    if 1 <= y <= 5:
        monthly[y-1] = 0.0
    elif 6 <= y <= 10:
        monthly[y-1] = 50.0
    elif 11 <= y <= 20:
        monthly[y-1] = 100.0
    else:
        monthly[y-1] = 150.0
annual_contrib = monthly * 12.0
years = np.arange(1, horizon+1)

# -----------------------
# Deterministic simulation (expected path)
# For FX we compound the fx_drift deterministically if MC=0
# -----------------------
def deterministic_simulation():
    nav = np.zeros(horizon)
    invested_cum = np.zeros(horizon)
    fx_rate = np.ones(horizon) * fx_initial
    # deterministic fx path: annual compounding of drift
    for t in range(horizon):
        if t == 0:
            fx_rate[t] = fx_initial * (1.0 + fx_drift)
        else:
            fx_rate[t] = fx_rate[t-1] * (1.0 + fx_drift)
    # start with initial capital invested at t=0 (we'll consider initial capital as already invested)
    total = 0.0
    invested = 0.0
    # we treat initial capital as invested at time 0
    invested += 0.0  # initial 1000 handled as base NAV?
    # We'll start NAV as initial capital (user earlier used 1000). Keep consistent and add initial 1000 invested.
    initial_capital = 1000.0
    total = initial_capital
    invested = initial_capital
    for t in range(horizon):
        # add annual contribution at the beginning of the year
        contrib = annual_contrib[t]
        total += contrib
        invested += contrib
        # compute portfolio return for the year:
        # For hedged ETFs, return in EUR is mu
        # For non-hedged ETFs, return in EUR approximated as (1+mu)*(1+fx_return)-1
        # Here fx_return = fx_rate[t]/fx_rate[t-1]-1 for t>0; for t==0 use fx_drift
        if t == 0:
            fx_ret = fx_drift
        else:
            fx_ret = fx_rate[t] / fx_rate[t-1] - 1.0
        # portfolio expected return this year
        yearly_returns = np.zeros_like(mus)
        for i in range(len(mus)):
            if hedged_flags[i]:
                yearly_returns[i] = mus[i]
            else:
                # approx combined effect
                yearly_returns[i] = (1.0 + mus[i]) * (1.0 + fx_ret) - 1.0
        port_mu = np.dot(alloc_norm, yearly_returns)
        total = total * (1.0 + port_mu)
        nav[t] = total
        invested_cum[t] = invested
    return nav, invested_cum, fx_rate

nav_det, invested_det, fx_det = deterministic_simulation()

# after-tax final: tax on gains at the end
final_nominal = nav_det[-1]
gain_nominal = final_nominal - invested_det[-1]
tax_amount = max(0.0, gain_nominal) * tax_rate
final_after_tax = final_nominal - tax_amount
# inflation adjustment (real value)
real_final = final_nominal / ((1.0 + inflation) ** horizon)
real_after_tax = final_after_tax / ((1.0 + inflation) ** horizon)

# -----------------------
# Monte Carlo simulation (if requested)
# Model annual returns per ETF ~ Normal(mu, sigma) (lognormal is alternative; this is a simplification)
# FX modeled as geometric (log returns normal) with drift fx_drift and vol fx_vol
# For each run simulate year-by-year, calculate NAV; accumulate final NAVs to compute percentiles & P(target)
# -----------------------
mc_results = None
mc_percentiles = None
p_target = None

if mc_runs > 0:
    rng = np.random.RandomState(None if mc_seed == 0 else int(mc_seed))
    finals = np.zeros(mc_runs)
    # For storing yearly percentiles optional
    all_paths = np.zeros((mc_runs, horizon))
    for r in range(mc_runs):
        total = 1000.0
        invested = 1000.0
        fx = fx_initial
        for t in range(horizon):
            # annual contribution at beginning
            contrib = annual_contrib[t]
            total += contrib
            invested += contrib
            # simulate ETF returns this year
            # draw normal shocks for each ETF
            shocks = rng.normal(loc=mus, scale=vols)
            # simulate fx log-return
            eps_fx = rng.normal(loc=fx_drift, scale=fx_vol)
            fx = fx * (1.0 + eps_fx)
            # combine returns for hedged / non-hedged
            yearly_returns = np.zeros_like(shocks)
            for i in range(len(shocks)):
                if hedged_flags[i]:
                    yearly_returns[i] = shocks[i]
                else:
                    yearly_returns[i] = (1.0 + shocks[i]) * (1.0 + eps_fx) - 1.0
            port_ret = np.dot(alloc_norm, yearly_returns)
            total = total * (1.0 + port_ret)
            all_paths[r, t] = total
        finals[r] = total
    mc_results = finals
    mc_percentiles = np.percentile(all_paths, [10, 25, 50, 75, 90], axis=0)  # shape (5, horizon)
    if target_active:
        p_target = np.mean(finals >= target_value)

# -----------------------
# Presentazione: grafici & tabella
# -----------------------
st.subheader("Risultato deterministico (atteso)")

colA, colB, colC, colD = st.columns(4)
colA.metric("Valore finale (nominale)", f"€ {final_nominal:,.0f}")
colB.metric("Totale investito", f"€ {invested_det[-1]:,.0f}")
colC.metric("Profitto (nominale)", f"€ {gain_nominal:,.0f}")
colD.metric("Valore netto dopo tasse", f"€ {final_after_tax:,.0f}")

colE, colF = st.columns(2)
colE.metric("Valore finale (reale, netto inflazione)", f"€ {real_final:,.0f}")
colF.metric("Valore netto dopo tasse (reale)", f"€ {real_after_tax:,.0f}")

if target_active:
    reached = final_after_tax >= target_value
    st.info(f"Target = € {target_value:,.0f}  →  Deterministic: {'Raggiunto' if reached else 'Non raggiunto'}")
    if mc_runs > 0:
        st.success(f"Monte Carlo: probabilità stimata di raggiungere target = {p_target*100:.1f}% ({mc_runs} simulazioni)")

# -----------------------
# Grafico principale
# -----------------------
st.subheader("Grafico: Capitale investito vs Valore del portafoglio")
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(years, invested_det, linestyle='--', label="Capitale investito cumulato", color="#2EC4B6", linewidth=2)
ax.plot(years, nav_det, label="Valore portafoglio (deterministico)", color="#FF9F1C", linewidth=3)

if mc_runs > 0:
    # shade percentiles area (10-90)
    p10 = mc_percentiles[0]
    p90 = mc_percentiles[-1]
    ax.fill_between(years, p10, p90, color="#8884ff", alpha=0.18, label="MC 10-90 percentile")
    # median
    ax.plot(years, mc_percentiles[2], color="#7a4cff", linestyle=':', label="MC mediana")

ax.set_title("Capitale investito vs Valore del portafoglio")
ax.set_xlabel("Anno")
ax.set_ylabel("Valore (€)")
ax.legend()
ax.grid(alpha=0.25)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

st.pyplot(fig)
plt.close(fig)

# -----------------------
# Tabella anno per anno (deterministica)
# -----------------------
st.subheader("Tabella anno per anno (deterministica)")
df = pd.DataFrame({
    "Anno": years,
    "Investito cumulato (€)": invested_det,
    "Valore nominale (€)": nav_det,
})
df["Profitto (€)"] = df["Valore nominale (€)"] - df["Investito cumulato (€)"]
df["Valore netto dopo tasse (€)"] = df["Valore nominale (€)"] - (np.maximum(0, df["Profitto (€)"]) * tax_rate)
df["Valore reale (€)"] = df["Valore nominale (€)"] / ((1.0 + inflation) ** df["Anno"].values)
st.dataframe(df.style.format("{:,.0f}"), height=360)

# CSV download
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Scarica tabella annuale (CSV)", data=csv, file_name="simulazione_anno_per_anno.csv", mime="text/csv")

# -----------------------
# Mostra FX path deterministico e primo MC sample if mc_runs>0
# -----------------------
st.subheader("Dettagli FX e distribuzioni (EUR/USD)")

st.write(f"FX iniziale impostato = {fx_initial:.3f}. Drift annuo impostato = {fx_drift:.3%}, vol = {fx_vol:.2%}")

if mc_runs > 0:
    st.write("Distribuzione dei valori finali (Monte Carlo)")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(mc_results, bins=40, color="#7a4cff", alpha=0.8)
    ax2.axvline(np.percentile(mc_results,50), color='k', linestyle='--', label='mediana')
    ax2.axvline(target_value, color='r', linestyle=':', label='target')
    ax2.set_xlabel("Valore finale (€)")
    ax2.set_ylabel("Frequenza")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.caption("Modello semplificato: gli annual returns sono campionati Normal(mu, sigma) per facilità educativa. Per analisi più sofisticate si possono usare processi log-normali e simulazioni con step mensili.")
