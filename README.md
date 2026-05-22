# 🎙️ BharatVaani — Indian Language Speech Translator

A **speech-to-speech translation system** for Indian languages — upload audio in any supported Indian language and get translated speech output, with **no need to specify the source language**. Built using billion-parameter transformer models and deployed on Hugging Face Spaces via Docker.

---

## 🌐 Live Demo

🔗 [https://prakruthisb-indian-speech-translator.hf.space/](https://prakruthisb-indian-speech-translator.hf.space/)

---

## 📌 Overview

Most translation tools require you to manually select the source language. BharatVaani eliminates this — it **automatically detects** the spoken language using Sarvam AI and routes the translation through the appropriate IndicTrans2 model, making it accessible for multilingual and low-literacy Indian users.

---

## ✨ Features

- 🔍 **Automatic language detection** — no need to specify the source language
- 🗣️ **8 Indian languages supported** — Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali, English
- 🔄 **Bidirectional translation** — any supported language ↔ any other
- 🎵 **Audio preprocessing** — mono conversion, 16kHz resampling, volume normalisation before ASR
- 🧠 **Billion-parameter translation models** — ai4bharat/IndicTrans2 1B with quantisation support
- 📥 **Download translated audio** — output MP3 downloadable directly from the UI
- 🐳 **Dockerised deployment** on Hugging Face Spaces

---

## 🔁 Pipeline

```
Audio Input (WAV / MP3 / MP4)
        │
        ▼
┌─────────────────────────┐
│   Audio Preprocessing   │  pydub + librosa
│  Mono · 16kHz · Normalise│
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│     Sarvam AI ASR       │  Speech → Text
│  + Language Detection   │  (auto-detected, no input needed)
└─────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│        IndicTrans2 Translation        │  ai4bharat 1B models
│  indic→en  /  en→indic  /  indic→indic│  routed by language pair
└──────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│     ElevenLabs TTS      │  eleven_multilingual_v2
│   Text → Speech Output  │
└─────────────────────────┘
        │
        ▼
   Output Audio (MP3)
```

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| Frontend | Streamlit |
| Speech-to-Text + Language Detection | Sarvam AI API |
| Translation Models | ai4bharat/IndicTrans2 1B (HuggingFace Transformers) |
| Translation Toolkit | IndicTransToolkit (IndicProcessor) |
| Text-to-Speech | ElevenLabs `eleven_multilingual_v2` |
| Audio Preprocessing | Pydub, Librosa, Soundfile |
| Model Optimisation | BitsAndBytesConfig (4-bit/8-bit quantisation), fp16, LRU cache |
| Containerisation | Docker |
| Deployment | Hugging Face Spaces |

---

## 🌍 Supported Languages

| Language | Code |
| --- | --- |
| Hindi | `hin_Deva` |
| Kannada | `kan_Knda` |
| Tamil | `tam_Taml` |
| Telugu | `tel_Telu` |
| Malayalam | `mal_Mlym` |
| Marathi | `mar_Deva` |
| Bengali | `ben_Beng` |
| English | `eng_Latn` |

---

## 🗂️ Project Structure

```
bharatvaani/
│
├── src/
│   ├── app.py                  # Streamlit frontend
    └── pipeline.py             # End-to-end translation pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/Prakruthisb/BharatVaani/tree/main
cd bharatvaani

# Set environment variables
export SARVAM_API_KEY=your_sarvam_key
export ELEVEN_API_KEY=your_elevenlabs_key
export HF_TOKEN=your_huggingface_token

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
```

### 🐳 Run with Docker

```bash
docker build -t bharatvaani .
docker run -p 8501:8501 \
  -e SARVAM_API_KEY=your_key \
  -e ELEVEN_API_KEY=your_key \
  -e HF_TOKEN=your_key \
  bharatvaani
```

App will be available at `http://localhost:8501`

---

## 🧠 Model Details

BharatVaani uses **3 separate IndicTrans2 1B-parameter models** from [ai4bharat](https://github.com/AI4Bharat/IndicTrans2), selected automatically based on the language pair:

| Task | Model |
| --- | --- |
| Indic → English | `ai4bharat/indictrans2-indic-en-1B` |
| English → Indic | `ai4bharat/indictrans2-en-indic-1B` |
| Indic → Indic | `ai4bharat/indictrans2-indic-indic-1B` |

**Memory optimisations applied:**
- LRU cache (`maxsize=3`) — models loaded once and reused
- 4-bit / 8-bit quantisation via `BitsAndBytesConfig` (optional)
- GPU fp16 half-precision when CUDA is available
- `low_cpu_mem_usage=True` during model loading
- Batch inference with beam search (`num_beams=5`, `batch_size=4`)

---

## 🔑 API Keys Required

| Service | Purpose | Get Key |
| --- | --- | --- |
| [Sarvam AI](https://sarvam.ai) | Speech-to-text + language detection | sarvam.ai |
| [ElevenLabs](https://elevenlabs.io) | Text-to-speech output | elevenlabs.io |
| [Hugging Face](https://huggingface.co) | IndicTrans2 model access | huggingface.co |

---

## 🔮 Future Improvements

- Real-time microphone input (no file upload needed)
- Add more Indian languages (Odia, Punjabi, Gujarati)
- Faster inference with ONNX / TorchScript export
- Speaker voice selection for TTS output

---

⭐ If you find this useful, consider giving it a star!