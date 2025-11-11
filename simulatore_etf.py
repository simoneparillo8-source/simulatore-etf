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
    page_title="Simulatore ETF Realistico",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Simulatore ETF — Dati realistici di mercato")
st.markdown(
    "Simula la crescita di un portafoglio di ETF reali (rendimenti e volatilità stimati). "
    "Puoi modificare **solo le allocazioni**, mantenendo dati coerenti con il mercato."
)

# ================================
# PARAMETRI BASE
# ================================
capitale_iniziale = 1000
versamento_6_10 = 50
versamento_11_20 = 100
versamento_21_40 = 150
anni = np.arange(1, 41)

st.sidebar.header("💰 Parametri di investimento")

# selettore anno
anno_finale = st.sidebar.slider("Anno di simulazione (1–40)", 1, 40, 20)

# ================================
# ETF REALISTICI
# ================================
etf_data = {
    "Vanguard FTSE All-World (VWCE)": {"return": 0.063, "vol": 0.135},
    "S&P 500 USD": {"return": 0.070, "vol": 0.140},
    "Europa / Globale": {"return": 0.055, "vol": 0.110},
    "IA / Tech": {"return": 0.085, "vol": 0.180},
    "Mercati Emergenti": {"return": 0.075, "vol": 0.160},
}

etf_names = list(etf_data.keys())

st.sidebar.markdown("### 📊 Allocazioni ETF")
active_etfs = {}
for name in etf_names:
    active = st.sidebar.checkbox(f"Includi {name}", True)
    alloc = st.sidebar.slider(f"Allocazione {name} (%)", 0, 100, 20 if active else 0)
    active_etfs[name] = {"alloc": alloc, "active": active}

# normalizza solo gli ETF attivi
alloc_tot = sum(v["alloc"] for v in active_etfs.values() if v["active"])
if alloc_tot == 0:
    st.error("⚠️ Imposta almeno un'allocazione maggiore di 0%.")
    st.stop()

alloc_norm = {n: v["alloc"] / alloc_tot for n, v in active_etfs.items() if v["active"]}

# ================================
# SIMULAZIONE
# ================================
valori = [capitale_iniziale]
for anno in anni:
    last = valori[-1]
    if 6 <= anno <= 10:
        last += versamento_6_10 * 12
    elif 11 <= anno <= 20:
        last += versamento_11_20 * 12
    elif anno >= 21:
        last += versamento_21_40 * 12

    crescita = sum(etf_data[n]["return"] * w for n, w in alloc_norm.items())
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
colors = plt.cm.Paired(np.linspace(0, 1, len(alloc_norm)))
ax_pie.pie(alloc_norm.values(),
           labels=[f"{n} ({w*100:.1f}%)" for n, w in alloc_norm.items()],
           colors=colors, autopct="%1.1f%%", startangle=90, textprops={'color':"w"})
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

st.markdown("#### ETF inclusi:")
for n, w in alloc_norm.items():
    r = etf_data[n]["return"] * 100
    v = etf_data[n]["vol"] * 100
    st.write(f"• {n} — rendimento medio: {r:.2f}%, volatilità: {v:.1f}%, peso: {w*100:.1f}%")

# ================================
# PDF EXPORT
# ================================
def genera_pdf():
    buffer_pdf = BytesIO()
    doc = SimpleDocTemplate(buffer_pdf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>📊 Report Portafoglio ETF - Realistico</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Orizzonte simulato: {anno_finale} anni", styles["Normal"]))
    story.append(Paragraph(f"Totale finale: {totale_finale:,.0f} €", styles["Normal"]))
    story.append(Paragraph(f"CAGR medio: {cagr:.2f}%", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Image(img_buf_main, width=450, height=250))
    story.append(Spacer(1, 12))
    story.append(Image(pie_buf, width=300, height=300))
    story.append(Spacer(1, 12))

    for n, w in alloc_norm.items():
        r = etf_data[n]["return"] * 100
        v = etf_data[n]["vol"] * 100
        story.append(Paragraph(f"{n}: rendimento {r:.2f}%, volatilità {v:.1f}%, allocazione {w*100:.1f}%", styles["Normal"]))

    doc.build(story)
    buffer_pdf.seek(0)
    return buffer_pdf

st.download_button(
    label="📄 Esporta Report PDF",
    data=genera_pdf(),
    file_name="report_portafoglio_reale.pdf",
    mime="application/pdf"
)
