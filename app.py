import streamlit as st
import os
from pipeline import translation

st.set_page_config(page_title="Speech Translator", layout="centered")

st.title("🎙️ Indian Language Speech Translator")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "mp4"])

lang_map_ui = {
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
    "English": "eng_Latn"
}

target_lang = st.selectbox("Select Target Language", list(lang_map_ui.keys()))

if uploaded_file:
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Translate 🎯"):
        with st.spinner("Processing..."):
            output_audio = translation("temp_audio", lang_map_ui[target_lang])

        st.success("Done!")

        st.audio(output_audio)

        with open(output_audio, "rb") as f:
            st.download_button(
                label="Download Output",
                data=f,
                file_name="translated.mp3",
                mime="audio/mp3"
            )
