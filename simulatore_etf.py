import streamlit as st
with col5:
new_day = st.selectbox("Giorno", ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"])


if st.button("Aggiungi"):
new_row = {
"Esercizio": new_name,
"Peso (kg)": new_weight,
"Ripetizioni": new_reps,
"Serie": new_sets,
"Giorno": new_day
}
user_df = pd.concat([user_df, pd.DataFrame([new_row])], ignore_index=True)
save_user_data(selected_user, user_df.to_dict(orient="records"))
st.experimental_rerun()


# ----------------------
# EDIT TABELLA
# ----------------------


st.subheader("📋 I tuoi allenamenti")
edited_df = st.data_editor(user_df, num_rows="dynamic", use_container_width=True)


# Salvataggio dopo modifica
save_user_data(selected_user, edited_df.to_dict(orient="records"))


# ----------------------
# ELIMINA ESERCIZIO
# ----------------------


st.subheader("🗑️ Rimuovi esercizio")
if len(edited_df) > 0:
to_delete = st.selectbox("Seleziona esercizio da eliminare", edited_df["Esercizio"].tolist())
if st.button("Elimina definitivamente"):
edited_df = edited_df[edited_df["Esercizio"] != to_delete]
save_user_data(selected_user, edited_df.to_dict(orient="records"))
st.experimental_rerun()


# ----------------------
# SLIDER VELOCI PER PESI
# ----------------------


st.subheader("🎚️ Regola velocemente i pesi")
for i, row in edited_df.iterrows():
new_val = st.slider(
f"{row['Esercizio']} - Peso (kg)", 0, 300, int(row['Peso (kg)'])
)
edited_df.at[i, "Peso (kg)"] = new_val


save_user_data(selected_user, edited_df.to_dict(orient="records"))


# ----------------------
# FRASE MOTIVAZIONALE
# ----------------------


st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Motivazione del giorno")
phrases = [
"Spingi oltre i tuoi limiti!",
"Ogni ripetizione conta.",
"La costanza vince sempre.",
"La versione migliore di te inizia oggi.",
]
st.sidebar.write(f"**{phrases[len(edited_df) % len(phrases)]}**")

