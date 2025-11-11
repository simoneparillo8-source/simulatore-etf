import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURAZIONE APP ===
st.set_page_config(page_title="Simulatore ETF Avanzato", layout="wide")
plt.style.use("dark_background")

# === PARAMETRI BASE ===
years = np.arange(1, 21)
initial_capital = 1000
monthly_contrib = [0]*5 + [50]*5 + [100]*10
annual_contrib = np.array(monthly_contrib) * 12
etf_names = ["S&P500 Hedged", "S&P500 Non Hedged", "Europa", "IA / Tech"]

# === RENDIMENTI PER SCENARIO ===
returns_scenarios = {
    "Pessimistico": [0.02, 0.025, 0.015, 0.03],
    "Medio": [0.052, 0.061, 0.045, 0.065],
    "Ottimistico": [0.075, 0.082, 0.065, 0.09]
}

# === UI ===
st.title("📈 Simulatore ETF — 3 Scenari (Dark Mode)")
st.markdown(
    "Simula la crescita del tuo portafoglio ETF su 20 anni. "
    "Puoi modificare le allocazioni e confrontare tre scenari: **pessimistico**, **medio** e **ottimistico**."
)

# --- SCELTA SCENARIO ---
scenario = st.sidebar.selectbox("Scegli scenario", list(returns_scenarios.keys()))

# --- SLIDER PER ALLOCAZIONI ---
st.sidebar.header("💼 Impostazioni di allocazione")
alloc = []
default_alloc = [0.4, 0.2, 0.25, 0.15]
for i, name in enumerate(etf_names):
    alloc.append(st.sidebar.slider(f"{name}", 0.0, 1.0, default_alloc[i], 0.05))
alloc = np.array(alloc)
alloc /= alloc.sum()  # Normalizza a 100%

st.sidebar.write(f"Totale allocazioni: **{alloc.sum()*100:.1f}%**")

# === FUNZIONE DI SIMULAZIONE ===
def simulate(returns):
    values = np.zeros((len(years), len(etf_names)))
    for i in range(len(etf_names)):
        val = initial_capital * alloc[i]
        for y in range(len(years)):
            val = val * (1 + returns[i]) + annual_contrib[y] * alloc[i]
            values[y, i] = val
    return values.sum(axis=1)

# === CALCOLA TUTTI GLI SCENARI ===
results = {}
for scen, r in returns_scenarios.items():
    results[scen] = simulate(r)

# === GRAFICO ===
fig, ax = plt.subplots(figsize=(10,6))
colors = {"Pessimistico":"#FF4B4B", "Medio":"#4BC0C0", "Ottimistico":"#00FF85"}

for scen, vals in results.items():
    ax.plot(years, vals, label=f"{scen} (finale: €{vals[-1]:,.0f})", linewidth=2.5, color=colors[scen])

ax.set_title("📊 Evoluzione del Portafoglio — Scenari comparati", fontsize=14)
ax.set_xlabel("Anno")
ax.set_ylabel("Valore (€)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# === RISULTATI FINALI ===
st.subheader(f"📘 Riepilogo scenario: {scenario}")
values = results[scenario]
final_value = values[-1]
total_invested = initial_capital + annual_contrib.sum()
gain = final_value - total_invested
cagr = (final_value / total_invested)**(1/20) - 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("Valore finale", f"€ {final_value:,.2f}")
col2.metric("Totale investito", f"€ {total_invested:,.2f}")
col3.metric("Guadagno", f"€ {gain:,.2f}")
col4.metric("Rendimento medio annuo (CAGR)", f"{cagr*100:.2f}%")

# === DETTAGLIO ETF ===
st.write("### Dettaglio ETF")
for i, name in enumerate(etf_names):
    st.write(f"- **{name}** — rendimento scenario *{scenario}*: {returns_scenarios[scenario][i]*100:.2f}%, allocazione: {alloc[i]*100:.1f}%")
