import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulatore ETF", layout="wide")

# --- PARAMETRI BASE ---
years = np.arange(1, 21)
initial_capital = 1000
monthly_contrib = [0]*5 + [50]*5 + [100]*10
annual_contrib = np.array(monthly_contrib) * 12

etf_names = ["S&P500 Hedged", "S&P500 Non Hedged", "Europa", "IA / Tech"]
returns = [0.052, 0.061, 0.045, 0.065]

st.title("📈 Simulatore Portafoglio ETF Interattivo")
st.markdown("Simula la crescita del tuo portafoglio ETF su 20 anni modificando le allocazioni con gli slider.")

# --- SLIDER PER ALLOCAZIONI ---
st.sidebar.header("🔧 Impostazioni di allocazione")
alloc = []
for i, name in enumerate(etf_names):
    alloc.append(st.sidebar.slider(f"{name}", 0.0, 1.0, [0.4, 0.2, 0.25, 0.15][i], 0.05))
alloc = np.array(alloc)
alloc /= alloc.sum()  # Normalizza a 100%

st.sidebar.write(f"**Totale allocazioni:** {alloc.sum()*100:.1f}% (normalizzato automaticamente)")

# --- SIMULAZIONE ---
values = np.zeros((len(years), len(etf_names)))
for i in range(len(etf_names)):
    val = initial_capital * alloc[i]
    for y in range(len(years)):
        val = val * (1 + returns[i]) + annual_contrib[y] * alloc[i]
        values[y, i] = val
total = values.sum(axis=1)

# --- GRAFICO ---
fig, ax = plt.subplots(figsize=(10,6))
for i in range(len(etf_names)):
    ax.plot(years, values[:, i], label=f"{etf_names[i]} ({alloc[i]*100:.1f}%)", alpha=0.7)
ax.plot(years, total, color="black", linewidth=2.5, label="Totale Portafoglio")
ax.set_title("Evoluzione del Portafoglio (20 anni)")
ax.set_xlabel("Anno")
ax.set_ylabel("Valore (€)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- RISULTATI FINALI ---
st.subheader("📊 Riepilogo finale")
final_value = total[-1]
total_invested = initial_capital + annual_contrib.sum()
gain = final_value - total_invested
cagr = (final_value / total_invested)**(1/20) - 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("Valore finale", f"€ {final_value:,.2f}")
col2.metric("Totale investito", f"€ {total_invested:,.2f}")
col3.metric("Guadagno totale", f"€ {gain:,.2f}")
col4.metric("Rendimento medio annuo (CAGR)", f"{cagr*100:.2f}%")

# --- DETTAGLIO ETF ---
st.write("### 📘 Dettaglio ETF")
for i, name in enumerate(etf_names):
    st.write(f"- **{name}**: rendimento {returns[i]*100:.2f}%, allocazione {alloc[i]*100:.1f}%")
