import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ================================
# CONFIGURAZIONE BASE STREAMLIT
# ================================
st.set_page_config(
    page_title="Simulatore ETF Portfolio",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Simulatore Portafoglio ETF — Lungo Periodo")
st.markdown(
    "Simula la crescita del tuo portafoglio ETF nel lungo periodo (10–20 anni). \
     Usa gli slider per modificare le allocazioni e osserva l’impatto sui rendimenti."
)

# ================================
# PARAMETRI BASE
# ================================
anni = np.arange(1, 21)
capitale_iniziale = 1000
versamento_6_10 = 50
versamento_11_20 = 100

st.sidebar.header("💰 Parametri di base")
alloc_sp500 = st.sidebar.slider("S&P 500 Hedged (%)", 0, 100, 35)
alloc_sp500usd = st.sidebar.slider("S&P 500 Non Hedged (%)", 0, 100, 15)
alloc_europe = st.sidebar.slider("Europa / Globale (%)", 0, 100, 35)
alloc_ai = st.sidebar.slider("Tecnologia / IA (%)", 0, 100, 15)

# normalizza a 100%
alloc_sum = alloc_sp500 + alloc_sp500usd + alloc_europe + alloc_ai
alloc_norm = np.array([alloc_sp500, alloc_sp500usd, alloc_europe, alloc_ai]) / alloc_sum

# ETF & rendimenti stimati (netti di TER e tasse)
etf_names = ["S&P 500 Hedged", "S&P 500 USD", "Europa / Globale", "IA / Tech"]
etf_returns = np.array([0.065, 0.07, 0.055, 0.085])
etf_vol = np.array([0.13, 0.14, 0.11, 0.18])

# Simulazione crescita
valori = [capitale_iniziale]
for anno in anni:
    last = valori[-1]
    # versamento periodico
    if anno > 5 and anno <= 10:
        last += versamento_6_10 * 12
    elif anno > 10:
        last += versamento_11_20 * 12
    # crescita ponderata
    crescita = np.sum(etf_returns * alloc_norm)
    last *= (1 + crescita)
    valori.append(last)

valori = valori[1:]
totale_finale = valori[-1]
rendimento_tot = (totale_finale / capitale_iniziale - 1) * 100
cagr = ((totale_finale / capitale_iniziale) ** (1 / len(anni)) - 1) * 100

# ================================
# GRAFICO PRINCIPALE
# ================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(anni, valori, linewidth=3, color="#00FF85", label="Portafoglio simulato")
ax.fill_between(anni, np.array(valori) * 0.95, np.array(valori) * 1.05, color="#00FF85", alpha=0.1)
ax.set_title("📈 Crescita del portafoglio nel tempo", fontsize=16, color="#00FF85")
ax.set_xlabel("Anno", color="w")
ax.set_ylabel("Valore (€)", color="w")
ax.legend(facecolor="#111", edgecolor="#333", labelcolor="w")
ax.grid(alpha=0.3)
fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")
ax.tick_params(colors="w")

# salva in buffer
img_buf_main = BytesIO()
fig.savefig(img_buf_main, format='png', bbox_inches='tight', dpi=200)
img_buf_main.seek(0)
plt.close('all')

# Mostra in Streamlit
st.image(img_buf_main, caption="📊 Andamento del portafoglio simulato", use_container_width=True)

# ================================
# PIE CHART allocazione
# ================================
fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
colors_pie = ["#00FF85", "#4B8BFF", "#FFB84C", "#B066FF"]
ax_pie.pie(alloc_norm, labels=[f"{n} ({a*100:.1f}%)" for n, a in zip(etf_names, alloc_norm)],
           colors=colors_pie, autopct="%1.1f%%", startangle=90, textprops={'color':"w"})
ax_pie.set_title("Allocazione ETF", color="w")
fig_pie.patch.set_facecolor("#0E1117")
ax_pie.set_facecolor("#0E1117")

pie_buf = BytesIO()
fig_pie.savefig(pie_buf, format='png', bbox_inches='tight', dpi=200)
pie_buf.seek(0)
plt.close('all')

col1, col2 = st.columns([1, 1])
col1.image(pie_buf, caption="💼 Distribuzione attuale", use_container_width=True)

# ================================
# INDICATORI PRINCIPALI
# ================================
st.markdown("### 📘 Riepilogo risultati")
st.write(f"**Totale finale (20 anni):** {totale_finale:,.0f} €")
st.write(f"**Rendimento complessivo:** {rendimento_tot:.2f}%")
st.write(f"**CAGR (rendimento medio annuo composto):** {cagr:.2f}%")

# ================================
# GENERAZIONE PDF REPORT
# ================================
def genera_pdf():
    buffer_pdf = BytesIO()
    doc = SimpleDocTemplate(buffer_pdf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>📊 Report Portafoglio ETF</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Totale finale: {totale_finale:,.0f} €", styles["Normal"]))
    story.append(Paragraph(f"CAGR medio: {cagr:.2f}%", styles["Normal"]))
    story.append(Spacer(1, 12))

    # grafico principale
    story.append(Image(img_buf_main, width=450, height=250))
    story.append(Spacer(1, 12))

    # pie chart
    story.append(Paragraph("<b>Allocazione ETF</b>", styles["Heading2"]))
    story.append(Image(pie_buf, width=300, height=300))
    story.append(Spacer(1, 12))

    # dettagli
    for i, name in enumerate(etf_names):
        story.append(Paragraph(
            f"{name}: rendimento {etf_returns[i]*100:.2f}%, volatilità {etf_vol[i]*100:.1f}%, allocazione {alloc_norm[i]*100:.1f}%",
            styles["Normal"]
        ))

    doc.build(story)
    buffer_pdf.seek(0)
    return buffer_pdf

# pulsante PDF
st.download_button(
    label="📄 Esporta Report PDF",
    data=genera_pdf(),
    file_name="report_portafoglio_etf.pdf",
    mime="application/pdf"
)
