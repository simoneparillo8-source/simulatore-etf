import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

# === CONFIGURAZIONE APP ===
st.set_page_config(page_title="Simulatore ETF — Report Avanzato", layout="wide")
plt.style.use("dark_background")

# === PARAMETRI DI BASE ===
years = np.arange(1, 21)
initial_capital = 1000
monthly_contrib = [0]*5 + [50]*5 + [100]*10
annual_contrib = np.array(monthly_contrib) * 12

# === ETF ===
etf_globale = ["S&P500 Non Hedged", "Vanguard FTSE All-World (VWCE)", "Europa (Vanguard)", "IA / Tech"]
returns_globale = [0.063, 0.060, 0.047, 0.080]
vol_globale = [0.14, 0.115, 0.105, 0.19]

etf_classico = ["S&P500 Non Hedged", "S&P500 Hedged", "Europa (Vanguard)", "IA / Tech"]
returns_classico = [0.061, 0.052, 0.045, 0.065]
vol_classico = [0.14, 0.10, 0.11, 0.18]

# === UI ===
st.title("📊 Simulatore ETF — Report Avanzato (Classico vs Globale)")
st.markdown("Analisi a 20 anni con **banda di volatilità**, **rendimento medio**, **drawdown stimato** e **commento automatico** del portafoglio.")

scenario = st.sidebar.selectbox("Scegli scenario di rendimento", ["Pessimistico", "Medio", "Ottimistico"])

# === SLIDER PER ALLOCAZIONI ===
st.sidebar.header("💼 Allocazioni ETF (Globale)")
default_alloc = [0.4, 0.25, 0.2, 0.15]
alloc = []
for i, name in enumerate(etf_globale):
    alloc.append(st.sidebar.slider(f"{name}", 0.0, 1.0, default_alloc[i], 0.05))
alloc = np.array(alloc)
alloc /= alloc.sum()

# === FUNZIONE DI SIMULAZIONE ===
def simulate(returns, vols):
    values = np.zeros(len(years))
    upper, lower = np.zeros(len(years)), np.zeros(len(years))
    total = initial_capital
    for y in range(len(years)):
        mean_r = np.dot(alloc, returns)
        sigma = np.sqrt(np.dot(alloc**2, np.array(vols)**2))
        total = total * (1 + mean_r) + annual_contrib[y]
        values[y] = total
        upper[y] = total * (1 + sigma)
        lower[y] = total * (1 - sigma)
    return values, upper, lower, sigma

factor = {"Pessimistico": 0.7, "Medio": 1.0, "Ottimistico": 1.2}[scenario]
returns_classico_adj = [r * factor for r in returns_classico]
returns_globale_adj = [r * factor for r in returns_globale]

# === SIMULAZIONI ===
values_c, upper_c, lower_c, sigma_c = simulate(returns_classico_adj, vol_classico)
values_g, upper_g, lower_g, sigma_g = simulate(returns_globale_adj, vol_globale)

# === GRAFICO ===
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(years, values_c, label="Classico (S&P500 Hedged)", color="#4B8BFF", linewidth=2)
ax.fill_between(years, lower_c, upper_c, color="#4B8BFF", alpha=0.15)
ax.plot(years, values_g, label="Globale (VWCE)", color="#00FF85", linewidth=2.5)
ax.fill_between(years, lower_g, upper_g, color="#00FF85", alpha=0.15)
ax.set_title(f"Evoluzione Portafoglio — Scenario {scenario}", fontsize=14)
ax.set_xlabel("Anno")
ax.set_ylabel("Valore (€)")
ax.legend()
ax.grid(True, alpha=0.3)

# Salva grafico in buffer
img_buf = BytesIO()
fig.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
img_buf.seek(0)
st.pyplot(fig)

# === METRICHE FINALI ===
total_invested = initial_capital + annual_contrib.sum()
final_c = values_c[-1]
final_g = values_g[-1]
cagr_c = (final_c / total_invested)**(1/20) - 1
cagr_g = (final_g / total_invested)**(1/20) - 1
drawdown_c = sigma_c * 2.2 * 100
drawdown_g = sigma_g * 2.2 * 100

# === COMMENTO AUTOMATICO ===
if cagr_g > cagr_c:
    perf_comment = f"Il portafoglio **Globale (VWCE)** mostra una crescita media superiore di **{(cagr_g-cagr_c)*100:.2f}%** annuo rispetto al Classico, grazie alla maggiore diversificazione geografica e valutaria."
else:
    perf_comment = f"Il portafoglio **Classico (S&P500 Hedged)** risulta leggermente più efficiente in questo scenario, con un rendimento medio superiore di **{(cagr_c-cagr_g)*100:.2f}%** annuo."
    
if drawdown_g < drawdown_c:
    risk_comment = f"In termini di rischio, il Globale presenta un drawdown stimato inferiore (**{drawdown_g:.1f}% vs {drawdown_c:.1f}%**), segno di minore volatilità media."
else:
    risk_comment = f"Il Classico appare più stabile con drawdown inferiore (**{drawdown_c:.1f}% vs {drawdown_g:.1f}%**), ma la differenza è contenuta."

time_comment = "L’orizzonte di 20 anni permette di sfruttare pienamente l’interesse composto, rendendo le oscillazioni intermedie meno rilevanti per un investitore giovane."

final_comment = f"{perf_comment} {risk_comment} {time_comment}"

# === FUNZIONE PER PDF ===
def genera_pdf():
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), title="Simulatore ETF Report Avanzato")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>📊 Simulatore ETF — Report Avanzato (20 anni)</b>", styles['Title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Scenario:</b> {scenario}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Image(img_buf, width=700, height=400))
    story.append(Spacer(1, 15))

    data = [
        ["Parametro", "Classico (S&P500 Hedged)", "Globale (VWCE)"],
        ["Valore finale (€)", f"{final_c:,.2f}", f"{final_g:,.2f}"],
        ["Totale investito (€)", f"{total_invested:,.2f}", f"{total_invested:,.2f}"],
        ["CAGR (%)", f"{cagr_c*100:.2f}", f"{cagr_g*100:.2f}"],
        ["Drawdown stimato (%)", f"{drawdown_c:.1f}", f"{drawdown_g:.1f}"]
    ]
    table = Table(data, hAlign='LEFT', colWidths=[200, 200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.gray),
        ('BACKGROUND', (0, 1), (-1, -1), colors.black),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>💬 Analisi automatica</b>", styles['Heading2']))
    story.append(Paragraph(final_comment, styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>💼 Dettaglio ETF (Portafoglio Globale)</b>", styles['Heading2']))
    for i, name in enumerate(etf_globale):
        story.append(Paragraph(f"{name}: rendimento {returns_globale_adj[i]*100:.2f}%, volatilità {vol_globale[i]*100:.1f}%, allocazione {alloc[i]*100:.1f}%", styles['Normal']))

    doc.build(story)
    buf.seek(0)
    return buf

# === INTERFACCIA STREAMLIT ===
st.subheader("📘 Confronto finale dopo 20 anni")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Valore finale Classico", f"€ {final_c:,.2f}")
col2.metric("Valore finale Globale", f"€ {final_g:,.2f}")
col3.metric("CAGR Classico", f"{cagr_c*100:.2f}%")
col4.metric("CAGR Globale", f"{cagr_g*100:.2f}%")

st.markdown(f"**Analisi sintetica:** {final_comment}")

# === PULSANTE PDF ===
if st.button("📄 Esporta PDF Report"):
    pdf_buf = genera_pdf()
    st.download_button(
        label="💾 Scarica Report PDF",
        data=pdf_buf,
        file_name="report_portafoglio_etf_avanzato.pdf",
        mime="application/pdf"
    )
