import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulatore ETF Globale (VWCE)", layout="wide")
plt.style.use("dark_background")

# === PARAMETRI BASE ===
years = np.arange(1, 21)
initial_capital = 1000
monthly_contrib = [0]*5 + [50]*5 + [100]*10
annual_contrib = np.array(monthly_contrib) * 12

# === ETF ===
etf_names = ["S&P500 Non Hedged", "Vanguard FTSE All-World (VWCE)", "Europa (Vanguard)", "IA / Tech"]

# === SCENARI DI RENDIMENTO ===
returns_scenarios = {
    "Pessimistico": [0.030, 0.027, 0.020, 0.035],
    "Medio": [0.063, 0.060, 0.047, 0.080],
    "Ottimistico": [0.080, 0.072, 0.060, 0.095]
}

# === UI ===
st.title("🌍 Simulatore ETF Globale — Portafoglio Giovane (20 anni)")
st.markdown(
    "Versione aggiornata con **Vanguard FTSE All-World (VWCE)** al posto dell'S&P500 Hedged. "
    "Confronta 3 scenari e regola le allocazioni con gli slider."
)

# --- SCELTA SCENARIO ---
scenario = st.sidebar.selectbox("Scegli scenario", list(returns_scenarios.keys()))

# --- SLIDER PER ALLOCAZIONI ---
st.sidebar.header("💼 Impostazioni di allocazione")
default_alloc = [0.4, 0.25, 0.2, 0.15]
alloc = []
for i, name in enumerate(etf_names):
    alloc.append(st.sidebar.slider(f"{name}", 0.0, 1.0, default_alloc[i], 0.05))
alloc = np.array(alloc)
alloc /= alloc.sum()  # Normalizza

st.sidebar.write(f"Totale allocazioni: **{alloc.sum()*100:.1f}%**")

# === FUNZIONE SIMULAZIONE ===
def simulate(returns):
    values = np.zeros((len(years), len(etf_names)))
    for i in range(len(etf_names)):
        val = initial_capital * alloc[i]
        for y in range(len(years)):
            val = val * (1 + returns[i]) + annual_contrib[y] * alloc[i]
            values[y, i] = val
    return values.sum(axis=1)

# === CALCOLA RISULTATI ===
results = {}
for scen, r in returns_scenarios.items():
    results[scen] = simulate(r)

# === GRAFICO ===
fig, ax = plt.subplots(figsize=(10,6))
colors = {"Pessimistico":"#FF4B4B", "Medio":"#4BC0C0", "Ottimistico":"#00FF85"}

for scen, vals in results.items():
    ax.plot(years, vals, label=f"{scen} (finale: €{vals[-1]:,.0f})", linewidth=2.5, color=colors[scen])

ax.set_title("📊 Evoluzione del Portafoglio Globale (20 anni)", fontsize=14)
ax.set_xlabel("Anno")
ax.set_ylabel("Valore (€)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# === RIEPILOGO ===
st.subheader(f"📘 Scenario selezionato: {scenario}")
values = results[scenario]
final_value = values[-1]
total_invested = initial_capital + annual_contrib.sum()
gain = final_value - total_invested
cagr = (final_value / total_invested)**(1/20) - 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("Valore finale", f"€ {final_value:,.2f}")
col2.metric("Totale investito", f"€ {total_invested:,.2f}")
col3.metric("Guadagno", f"€ {gain:,.2f}")
col4.metric("CAGR medio", f"{cagr*100:.2f}%")

# === DETTAGLIO ETF ===
st.write("### Dettaglio ETF e rendimenti per scenario")
for i, name in enumerate(etf_names):
    st.write(f"- **{name}** — rendimento {returns_scenarios[scenario][i]*100:.2f}% | allocazione: {alloc[i]*100:.1f}%")

