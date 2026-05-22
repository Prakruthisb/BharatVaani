import streamlit as st
import os
# from pipeline import translation, SARVAM_API_KEY, ELEVEN_API_KEY
from pipeline import translation

st.set_page_config(page_title="Speech Translator", layout="centered")

st.title("🎙️ Indian Language Speech Translator")

# st.write("SARVAM KEY:", SARVAM_API_KEY)
# st.write("ELEVEN KEY:", ELEVEN_API_KEY)

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "mp4"])

# st.write(uploaded_file)

if uploaded_file is not None:
    st.success("File uploaded successfully ✅")
    # st.write("File name:", uploaded_file.name)
# elif uploaded_file is None:
#     st.write('failure')
# else:
#     st.warning("Please upload a file")

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp_audio.wav")
    
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

if uploaded_file is not None:
    # with open("temp_audio.wav", "wb") as f:
    #     f.write(uploaded_file.read())

    # st.audio("temp_audio.wav")

    if st.button("Translate 🎯"):
        with st.spinner("Processing..."):
            try:
                output_audio = translation("temp_audio.wav", lang_map_ui[target_lang])
                st.success("Done!")
                st.audio(output_audio)

                with open(output_audio, "rb") as f:
                    st.download_button(
                        label="Download Output",
                        data=f,
                        file_name="translated.mp3",
                        mime="audio/mp3"
                    )

            except Exception as e:
                st.error(str(e))
