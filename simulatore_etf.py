import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ================================
# CONFIGURAZIONE BASE
# ================================
st.set_page_config(
    page_title="Simulatore ETF — fino a 40 anni",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Simulatore Portafoglio ETF — Orizzonte fino a 40 anni")
st.markdown(
    "Cambia le allocazioni, seleziona l'anno di simulazione e osserva come evolve il portafoglio nel tempo."
)

# ================================
# PARAMETRI DI BASE
# ================================
capitale_iniziale = 1000
versamento_6_10 = 50
versamento_11_20 = 100
versamento_21_40 = 150

anni = np.arange(1, 41)

st.sidebar.header("💰 Parametri di investimento")

# slider allocazioni
alloc_sp500 = st.sidebar.slider("S&P 500 Hedged (%)", 0, 100, 30)
alloc_sp500usd = st.sidebar.slider("S&P 500 USD (%)", 0, 100, 20)
alloc_europe = st.sidebar.slider("Europa / Globale (%)", 0, 100, 35)
alloc_ai = st.sidebar.slider("IA / Tech (%)", 0, 100, 15)

alloc_sum = alloc_sp500 + alloc_sp500usd + alloc_europe + alloc_ai
alloc_norm = np.array([alloc_sp500, alloc_sp500usd, alloc_europe, alloc_ai]) / alloc_sum

# selettore anno finale
anno_finale = st.sidebar.slider("Anno di simulazione (1–40)", 1, 40, 20)

# ETF
etf_names = ["S&P 500 Hedged", "S&P 500 USD", "Europa / Globale", "IA / Tech"]
etf_returns = np.array([0.065, 0.07, 0.055, 0.085])
etf_vol = np.array([0.13, 0.14, 0.11, 0.18])

# ================================
# SIMULAZIONE
# ================================
valori = [capitale_iniziale]
for anno in anni:
    last = valori[-1]
    # versamenti annuali
    if anno > 5 and anno <= 10:
        last += versamento_6_10 * 12
    elif anno > 10 and anno <= 20:
        last += versamento_11_20 * 12
    elif anno > 20:
        last += versamento_21_40 * 12
    # crescita
    crescita = np.sum(etf_returns * alloc_norm)
    last *= (1 + crescita)
    valori.append(last)

valori = np.array(valori[1:])
valori_vis = valori[:anno_finale]

totale_finale = valori_vis[-1]
rendimento_tot = (totale_finale / capitale_iniziale - 1) * 100
cagr = ((totale_finale / capitale_iniziale) ** (1 / anno_finale) - 1) * 100

# ================================
# GRAFICO PRINCIPALE
# ================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, anno_finale + 1), valori_vis, linewidth=3, color="#00FF85", label="Portafoglio simulato")
ax.fill_between(np.arange(1, anno_finale + 1),
                valori_vis * 0.95, valori_vis * 1.05,
                color="#00FF85", alpha=0.1)
ax.set_title(f"📈 Crescita del portafoglio in {anno_finale} anni", fontsize=16, color="#00FF85")
ax.set_xlabel("Anno", color="w")
ax.set_ylabel("Valore (€)", color="w")
ax.legend(facecolor="#111", edgecolor="#333", labelcolor="w")
ax.grid(alpha=0.3)
fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")
ax.tick_params(colors="w")

img_buf_main = BytesIO()
fig.savefig(img_buf_main, format='png', bbox_inches='tight', dpi=200)
img_buf_main.seek(0)
plt.close('all')

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
# RISULTATI
# ================================
st.markdown("### 📘 Riepilogo risultati")
st.write(f"**Anno visualizzato:** {anno_finale}")
st.write(f"**Totale finale:** {totale_finale:,.0f} €")
st.write(f"**Rendimento complessivo:** {rendimento_tot:.2f}%")
st.write(f"**CAGR medio annuo:** {cagr:.2f}%")

# ================================
# PDF EXPORT
# ================================
def genera_pdf():
    buffer_pdf = BytesIO()
    doc = SimpleDocTemplate(buffer_pdf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>📊 Report Portafoglio ETF</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Orizzonte simulato: {anno_finale} anni", styles["Normal"]))
    story.append(Paragraph(f"Totale finale: {totale_finale:,.0f} €", styles["Normal"]))
    story.append(Paragraph(f"CAGR medio: {cagr:.2f}%", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Image(img_buf_main, width=450, height=250))
    story.append(Spacer(1, 12))
    story.append(Image(pie_buf, width=300, height=300))
    story.append(Spacer(1, 12))

    for i, name in enumerate(etf_names):
        story.append(Paragraph(
            f"{name}: rendimento {etf_returns[i]*100:.2f}%, "
            f"volatilità {etf_vol[i]*100:.1f}%, "
            f"allocazione {alloc_norm[i]*100:.1f}%",
            styles["Normal"]
        ))

    doc.build(story)
    buffer_pdf.seek(0)
    return buffer_pdf

st.download_button(
    label="📄 Esporta Report PDF",
    data=genera_pdf(),
    file_name="report_portafoglio_etf_40anni.pdf",
    mime="application/pdf"
)
