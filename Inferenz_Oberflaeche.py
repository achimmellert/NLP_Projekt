import os.path
import streamlit as st
import joblib
from pathlib import Path
from pandas import DataFrame
from model import SentenceBERTVectorizer


MODEL_PATH = Path.cwd() / "Models" / "Streamlit" / "streamlit_model.joblib"


st.set_page_config(
    page_title="Review Vorhersage",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Bewertungsvorhersage für Titel & Text")
st.snow()
st.markdown("Gib einen **Titel** und einen **Textinhalt** ein, um die zugehörige Bewertung vorherzusagen.")


session_keys = ["model"]

if all(key not in st.session_state.keys() for key in session_keys):
    st.session_state["model"] = []

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modell nicht gefunden unter: {MODEL_PATH}")
        st.stop()
    try:
        with st.spinner(show_time=True, text="Loading Model..."):
            nlp_pipeline = joblib.load(filename=MODEL_PATH)
            return nlp_pipeline, nlp_pipeline.classes_.tolist()
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.stop()

nlp_pipeline, classes = load_model()
st.session_state["classes"] = classes

with st.form():
    title = st.text_input("Titel", placeholder="Z.B. 'Great Service'")
    body = st.text_area("Text", placeholder="Z.B. 'The product exceeded my expectations...'")

    submit = st.form_submit_button("Bewertung vorhersagen")

if submit:
    if not title.strip() or not body.strip():
        st.warning("Bitte gib Titel und Text ein")
    else:
        input_data = f"{title} {body}"
        input_df = DataFrame(input_data, index=[0])

        try:
            with st.spinner(show_time=True):
                proba = nlp_pipeline.predict_proba(input_df)[0]

            best_idx = proba.argmax()
            predicted_class = st.session_state["classes"][best_idx]
            confidence = proba[best_idx]

            st.success(f"**Vorhergesagte Bewertung:** {predicted_class}")
            st.info(f"**Konfidenz:** {confidence:.2%}")
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")
else:
    st.info("Klicke auf den Button, sobald du Titel und Text eingegeben hast.")
