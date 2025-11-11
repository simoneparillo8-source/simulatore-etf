# simulatore_etf_pdf_finale_fixed.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Flowable

# === CONFIG ===
st.set_page_config(page_title="Simulatore ETF — Report Avanzato (FIXED)", layout="wide")
plt.style.use("dark_background")

# === PARAMETRI ===
years = np.arange(1, 21)
initial_capital = 1000.0
monthly_contrib = [0]*5 + [50]*5 + [100]*10
annual_contrib = np.array(monthly_contrib) * 12.0

# === ETF LISTE (nomi statici) ===
etf_globale = ["S&P500 Non Hedged", "Vanguard FTSE All-World (VWCE)", "Europa (Vanguard)", "IA / Tech"]
etf_classico = ["S&P500 Non Hedged", "S&P500 Hedged", "Europa (Vanguard)", "IA / Tech"]

# === RENDIMENTI BASE E VOLATILITÀ BASE (per scenario verranno scalati) ===
returns_globale = [0.063, 0.060, 0.047, 0.080]
vol_globale = [0.14, 0.115, 0.105, 0.19]

returns_classico = [0.061, 0.052, 0.045, 0.065]
vol_classico = [0.14, 0.10, 0.11, 0.18]

# === UI ===
st.title("📊 Simulatore ETF — Report Avanzato (FIXED)")
st.markdown("Confronto Classico vs Globale, bande di volatilità, drawdown stimato, e PDF export orizzontale.")

scenario = st.sidebar.selectbox("Scegli scenario di rendimento", ["Pessimistico", "Medio", "Ottimistico"])

# Allocazioni: separiamo i due portafogli così puoi confrontare e modificare
st.sidebar.header("💼 Allocazioni — Portafoglio Globale (VWCE)")
default_alloc_glob = [0.4, 0.25, 0.2, 0.15]
alloc_glob = []
for i, name in enumerate(etf_globale):
    alloc_glob.append(st.sidebar.slider(f"Globale - {name}", 0.0, 1.0, default_alloc_glob[i], 0.05))
alloc_glob = np.array(alloc_glob)
if alloc_glob.sum() == 0:
    alloc_glob = np.array(default_alloc_glob)
else:
    alloc_glob = alloc_glob / alloc_glob.sum()

st.sidebar.header("💼 Allocazioni — Portafoglio Classico (Hedged)")
default_alloc_class = [0.4, 0.2, 0.25, 0.15]
alloc_class = []
for i, name in enumerate(etf_classico):
    alloc_class.append(st.sidebar.slider(f"Classico - {name}", 0.0, 1.0, default_alloc_class[i], 0.05))
alloc_class = np.array(alloc_class)
if alloc_class.sum() == 0:
    alloc_class = np.array(default_alloc_class)
else:
    alloc_class = alloc_class / alloc_class.sum()

# === Scenario scaling factor ===
factor_map = {"Pessimistico": 0.7, "Medio": 1.0, "Ottimistico": 1.2}
factor = factor_map.get(scenario, 1.0)

returns_globale_adj = [r * factor for r in returns_globale]
returns_classico_adj = [r * factor for r in returns_classico]

# === funzione di simulazione (prende allocazione come argomento) ===
def simulate_with_alloc(returns, vols, alloc):
    years_len = len(years)
    values = np.zeros(years_len)
    upper = np.zeros(years_len)
    lower = np.zeros(years_len)
    total = initial_capital
    for y in range(years_len):
        # rendimento medio portafoglio (ponderato)
        mean_r = float(np.dot(alloc, returns))
        # volatilità portafoglio (approssimazione: radice somma pesata dei quadrati)
        sigma = float(np.sqrt(np.dot((alloc**2), (np.array(vols)**2))))
        # applichiamo rendimento/versamento annuale (versamento all'inizio dell'anno)
        total = total * (1.0 + mean_r) + annual_contrib[y]
        values[y] = total
        upper[y] = total * (1.0 + sigma)
        lower[y] = total * (1.0 - sigma)
    return values, upper, lower, sigma

# === Simulazioni per entrambi i portafogli ===
values_g, upper_g, lower_g, sigma_g = simulate_with_alloc(returns_globale_adj, vol_globale, alloc_glob)
values_c, upper_c, lower_c, sigma_c = simulate_with_alloc(returns_classico_adj, vol_classico, alloc_class)

# === Grafico principale ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years, values_c, label="Classico (S&P500 Hedged)", color="#4B8BFF", linewidth=2)
ax.fill_between(years, lower_c, upper_c, color="#4B8BFF", alpha=0.15)
ax.plot(years, values_g, label="Globale (VWCE)", color="#00FF85", linewidth=2.5)
ax.fill_between(years, lower_g, upper_g, color="#00FF85", alpha=0.15)
ax.set_title(f"Evoluzione Portafoglio — Scenario {scenario}", fontsize=14)
ax.set_xlabel("Anno")
ax.set_ylabel("Valore (€)")
ax.legend()
ax.grid(True, alpha=0.3)

# Salva immagine principale in buffer
img_buf_main = BytesIO()
fig.savefig(img_buf_main, format="png", dpi=200, bbox_inches="tight")
img_buf_main.seek(0)
plt.close(fig)  # chiudi figura per evitare warning/memory leak

# mostra in Streamlit
st.pyplot(plt.imread(img_buf_main))  # rapido rendering dell'immagine

# === Metriche finali ===
total_invested = initial_capital + annual_contrib.sum()
final_g = float(values_g[-1])
final_c = float(values_c[-1])
cagr_g = (final_g / total_invested)**(1/20) - 1
cagr_c = (final_c / total_invested)**(1/20) - 1
drawdown_g = sigma_g * 2.2 * 100
drawdown_c = sigma_c * 2.2 * 100

st.subheader("📘 Confronto finale dopo 20 anni")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Valore finale Globale", f"€ {final_g:,.2f}")
col2.metric("Valore finale Classico", f"€ {final_c:,.2f}")
col3.metric("CAGR Globale", f"{cagr_g*100:.2f}%")
col4.metric("CAGR Classico", f"{cagr_c*100:.2f}%")

st.write(f"Drawdown stimato (Globale): ≈ {drawdown_g:.1f}% — Drawdown stimato (Classico): ≈ {drawdown_c:.1f}%")

# === Funzione per creare pie chart in buffer (per PDF) ===
def make_pie_buf(labels, sizes, colors_pie, title=None):
    figp, axp = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = axp.pie(sizes, labels=labels, colors=colors_pie, startangle=90,
                                       autopct="%1.1f%%", textprops={'color': "w"})
    if title:
        axp.set_title(title, color="w")
    # assicuriamo contrasto testo
    for t in texts + autotexts:
        t.set_fontsize(8)
    buf = BytesIO()
    figp.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=figp.get_facecolor())
    buf.seek(0)
    plt.close(figp)
    return buf

# --- crea pie buffers per entrambi i portafogli
labels_g = [f"{n} ({alloc_glob[i]*100:.1f}%)" for i, n in enumerate(etf_globale)]
colors_pie = ["#00FF85", "#4B8BFF", "#FFB84C", "#B066FF"]
pie_buf_glob = make_pie_buf(labels_g, alloc_glob, colors_pie, title="Allocazione Globale")

labels_c = [f"{n} ({alloc_class[i]*100:.1f}%)" for i, n in enumerate(etf_classico)]
pie_buf_class = make_pie_buf(labels_c, alloc_class, colors_pie, title="Allocazione Classico")

# === Generazione PDF (landscape A4) ===
def genera_pdf():
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), title="Simulatore ETF Report Avanzato (FIXED)")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>📊 Simulatore ETF — Report Avanzato (20 anni)</b>", styles['Title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Scenario:</b> {scenario}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Inserisci immagine principale (grafico)
    # reportlab Image può ricevere un BytesIO oggetto
    story.append(Image(img_buf_main, width=700, height=380))
    story.append(Spacer(1, 12))

    # Tabella riassuntiva
    data = [
        ["Parametro", "Globale (VWCE)", "Classico (S&P500 Hedged)"],
        ["Valore finale (€)", f"{final_g:,.2f}", f"{final_c:,.2f}"],
        ["Totale investito (€)", f"{total_invested:,.2f}", f"{total_invested:,.2f}"],
        ["CAGR (%)", f"{cagr_g*100:.2f}", f"{cagr_c*100:.2f}"],
        ["Drawdown stimato (%)", f"{drawdown_g:.1f}", f"{drawdown_c:.1f}"]
    ]
    table = Table(data, hAlign='LEFT', colWidths=[240, 200, 200])
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
    story.append(Spacer(1, 12))

    # Affianca le due pie charts: stampiamo una dopo l'altra con spazio
    story.append(Paragraph("<b>💼 Allocazioni</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    # prima pie (globale)
    story.append(Image(pie_buf_glob, width=260, height=260))
    story.append(Spacer(1, 6))
    # poi pie (classico)
    story.append(Image(pie_buf_class, width=260, height=260))
    story.append(Spacer(1, 12))

    # Commento automatico
    if cagr_g > cagr_c:
        perf_comment = f"Il portafoglio Globale (VWCE) mostra una crescita media superiore di {(cagr_g-cagr_c)*100:.2f}% annuo rispetto al Classico, grazie alla diversificazione geografica."
    else:
        perf_comment = f"Il portafoglio Classico (S&P500 Hedged) mostra una crescita media superiore di {(cagr_c-cagr_g)*100:.2f}% annuo in questo scenario."

    if drawdown_g < drawdown_c:
        risk_comment = f"Il Globale presenta drawdown stimato inferiore ({drawdown_g:.1f}% vs {drawdown_c:.1f}%), suggerendo volatilità media leggermente minore."
    else:
        risk_comment = f"Il Classico presenta drawdown stimato inferiore ({drawdown_c:.1f}% vs {drawdown_g:.1f}%)."

    time_comment = "L'orizzonte di 20 anni riduce l'impatto delle oscillazioni a breve termine; la coerenza dei versamenti è più importante del market timing."

    story.append(Paragraph("<b>💬 Analisi automatica</b>", styles['Heading2']))
    story.append(Paragraph(perf_comment + " " + risk_comment + " " + time_comment, styles['Normal']))
    story.append(Spacer(1, 12))

    # Dettaglio ETF Globale (testo)
    story.append(Paragraph("<b>📘 Dettaglio ETF (Globale)</b>", styles['Heading2']))
    for i, name in enumerate(etf_globale):
        story.append(Paragraph(f"{name}: rendimento {returns_globale_adj[i]*100:.2f}%, volatilità {vol_globale[i]*100:.1f}%, allocazione {alloc_glob[i]*100:.1f}%", styles['Normal']))
        story.append(Spacer(1, 4))

    doc.build(story)
    buf.seek(0)
    return buf

# === Pulsante per esportare PDF ===
if st.button("📄 Esporta PDF Report"):
    try:
        pdf_buf = genera_pdf()
        st.download_button(
            label="💾 Scarica Report PDF",
            data=pdf_buf.getvalue(),
            file_name="report_portafoglio_etf_avanzato_fixed.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Errore generazione PDF: {e}")
        raise
