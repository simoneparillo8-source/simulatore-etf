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
    page_title="Simulatore ETF Avanzato",
    page_icon="💹",
    layout="wide",
)

st.title("💹 Simulatore ETF — Realistico, Moderno e Interattivo")
st.markdown("""
Benvenuto nel tuo simulatore finanziario personale!  
💼 Qui puoi testare scenari **pessimistici, neutri e ottimistici** con dati di mercato realistici.  
Guarda **quanto investi nel tempo**, quanto cresce il capitale e scarica un **report PDF** elegante.
""")

# ================================
# PARAMETRI BASE
# ================================
st.sidebar.title("⚙️ Controlli principali")

st.sidebar.subheader("💰 Parametri di investimento")
capitale_iniziale = 1000
versamento_6_10 = 50
versamento_11_20 = 100
versamento_21_40 = 150
anni = np.arange(1, 41)

anno_finale = st.sidebar.slider("Anno di simulazione", 1, 40, 20)

# ================================
# SCENARI
# ================================
st.sidebar.subheader("📈 Scenario di mercato")
scenario = st.sidebar.radio(
    "Seleziona uno scenario:",
    ("Pessimistico", "Neutro", "Ottimistico"),
    index=1
)

mult_rend = {"Pessimistico": 0.7, "Neutro": 1.0, "Ottimistico": 1.25}[scenario]
mult_vol = {"Pessimistico": 1.25, "Neutro": 1.0, "Ottimistico": 0.85}[scenario]

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

st.sidebar.subheader("📊 Allocazione ETF")
alloc = {}
for nome in etf_data.keys():
    active = st.sidebar.checkbox(f"Includi {nome}", True)
    percent = st.sidebar.slider(f"Allocazione {nome} (%)", 0, 100, 20 if active else 0)
    if active:
        alloc[nome] = percent

if sum(alloc.values()) == 0:
    st.error("⚠️ Devi selezionare almeno un ETF con percentuale > 0.")
    st.stop()

# normalizzazione
alloc_norm = {n: p / sum(alloc.values()) for n, p in alloc.items()}

# ================================
# SIMULAZIONE
# ================================
valore_tot = [capitale_iniziale]
capitale_investito = [capitale_iniziale]

for anno in anni:
    last = valore_tot[-1]
    investito = capitale_investito[-1]

    if 6 <= anno <= 10:
        investito += versamento_6_10 * 12
        last += versamento_6_10 * 12
    elif 11 <= anno <= 20:
        investito += versamento_11_20 * 12
        last += versamento_11_20 * 12
    elif anno >= 21:
        investito += versamento_21_40 * 12
        last += versamento_21_40 * 12

    crescita = sum(etf_data[n]["return"] * mult_rend * w for n, w in alloc_norm.items())
    last *= (1 + crescita)
    valore_tot.append(last)
    capitale_investito.append(investito)

valore_tot = np.array(valore_tot[1:])
capitale_investito = np.array(capitale_investito[1:])
anni_vis = anni[:anno_finale]

# ================================
# RISULTATI
# ================================
valore_finale = valore_tot[anno_finale - 1]
investito_finale = capitale_investito[anno_finale - 1]
profitto = valore_finale - investito_finale
rendimento_tot = (valore_finale / investito_finale - 1) * 100
cagr = ((valore_finale / investito_finale) ** (1 / anno_finale) - 1) * 100

st.markdown("## 💡 Riepilogo risultati")

col1, col2, col3 = st.columns(3)
col1.metric("Totale investito", f"{investito_finale:,.0f} €")
col2.metric("Valore portafoglio", f"{valore_finale:,.0f} €")
col3.metric("Profitto", f"{profitto:,.0f} €", f"{rendimento_tot:.2f}%")

st.markdown(f"📆 Periodo simulato: **{anno_finale} anni** — Scenario: **{scenario}**")
st.divider()

# ================================
# GRAFICO MODERNO
# ================================
st.markdown("## 📊 Evoluzione nel tempo")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(anni_vis, capitale_investito[:anno_finale], label="💶 Capitale Investito", color="#2EC4B6", linewidth=3, linestyle="--")
ax.plot(anni_vis, valore_tot[:anno_finale], label="📈 Valore Portafoglio", color="#FF9F1C", linewidth=4)
ax.fill_between(anni_vis, capitale_investito[:anno_finale], valore_tot[:anno_finale],
                where=valore_tot[:anno_finale] >= capitale_investito[:anno_finale],
                color="#2EC4B6", alpha=0.2)
ax.fill_between(anni_vis, capitale_investito[:anno_finale], valore_tot[:anno_finale],
                where=valore_tot[:anno_finale] < capitale_investito[:anno_finale],
                color="#E71D36", alpha=0.2)

ax.set_title(f"📊 Crescita del capitale nel tempo ({scenario})", fontsize=16, color="#F4F4F9")
ax.set_xlabel("Anno", color="#F4F4F9")
ax.set_ylabel("Valore (€)", color="#F4F4F9")
ax.legend(facecolor="#111", edgecolor="#333", labelcolor="w")
ax.grid(alpha=0.3)
fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")
ax.tick_params(colors="#F4F4F9")

img_buf_main = BytesIO()
fig.savefig(img_buf_main, format='png', bbox_inches='tight', dpi=200)
img_buf_main.seek(0)
plt.close('all')

st.image(img_buf_main, caption="📈 Confronto tra capitale investito e valore totale", use_container_width=True)

# ================================
# ALLOCAZIONE ETF
# ================================
st.markdown("## 💼 Distribuzione ETF")
fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(alloc_norm)))
ax_pie.pie(alloc_norm.values(),
           labels=[f"{n} ({p*100:.1f}%)" for n, p in alloc_norm.items()],
           colors=colors, autopct="%1.1f%%", startangle=90, textprops={'color': "w"})
ax_pie.set_title("Distribuzione attuale", color="w")
fig_pie.patch.set_facecolor("#0E1117")
ax_pie.set_facecolor("#0E1117")
pie_buf = BytesIO()
fig_pie.savefig(pie_buf, format='png', bbox_inches='tight', dpi=200)
pie_buf.seek(0)
plt.close('all')

col1, col2 = st.columns([1, 2])
col1.image(pie_buf, use_container_width=True)
col2.write("### Dettagli ETF attivi:")
for n, w in alloc_norm.items():
    r = etf_data[n]["return"] * mult_rend * 100
    v = etf_data[n]["vol"] * mult_vol * 100
    col2.write(f"• **{n}** — rendimento medio: {r:.2f}%, volatilità: {v:.1f}%, peso: {w*100:.1f}%")

# ================================
# PDF EXPORT
# ================================
def genera_pdf():
    buffer_pdf = BytesIO()
    doc = SimpleDocTemplate(buffer_pdf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>💹 Report Portafoglio ETF - Versione Moderna</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Scenario: {scenario}", styles["Normal"]))
    story.append(Paragraph(f"Totale investito: {investito_finale:,.0f} €", styles["Normal"]))
    story.append(Paragraph(f"Valore finale: {valore_finale:,.0f} €", styles["Normal"]))
    story.append(Paragraph(f"Profitto: {profitto:,.0f} €", styles["Normal"]))
    story.append(Paragraph(f"CAGR medio: {cagr:.2f}%", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Image(img_buf_main, width=450, height=250))
    story.append(Spacer(1, 12))
    story.append(Image(pie_buf, width=300, height=300))

    for n, w in alloc_norm.items():
        r = etf_data[n]["return"] * mult_rend * 100
        v = etf_data[n]["vol"] * mult_vol * 100
        story.append(Paragraph(f"{n}: rendimento {r:.2f}%, volatilità {v:.1f}%, allocazione {w*100:.1f}%", styles["Normal"]))

    doc.build(story)
    buffer_pdf.seek(0)
    return buffer_pdf

st.download_button(
    label="📄 Scarica Report PDF",
    data=genera_pdf(),
    file_name=f"portafoglio_moderno_{scenario.lower()}.pdf",
    mime="application/pdf"
)
