#audio preprocessing
from pydub import AudioSegment
import librosa
import soundfile as sf
def preprocess_audio(input_file, output_file="input2.wav"):
    # Step 1: Convert to mono + 16kHz
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print("AUDIO ERROR:", e)
        raise e
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    temp_file = "temp.wav"
    audio.export(temp_file, format="wav")

    # Step 2: Load audio for normalization
    y, sr = librosa.load(temp_file, sr=16000)

    # Step 3: Normalize volume
    y = librosa.util.normalize(y)

    # Step 4: Save final audio
    sf.write(output_file, y, sr)

    return output_file



import requests
import os

def speech_to_text(file_path):
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "Authorization": f'Bearer {SARVAM_API_KEY}'
    }

    audio_file = preprocess_audio(file_path)
    # audio_file = file_path

    with open(audio_file, "rb") as f:
        response = requests.post(
            url,
            headers=headers,
            files={"file": (audio_file, f, "audio/wav")},
            # files={"audio": (audio_file, f, "audio/wav")},
            data={}
        )
    
    if response.status_code != 200:
        raise Exception(f"Sarvam API Error {response.status_code}: {response.text}")
    
    # Convert JSON safely
    result = response.json()

    # Debug (optional)
    # print(result)

    transcript = result.get("transcript", "")
    language_code = result.get("language_code", "")

    return transcript, language_code
# print(speech_to_text("/content/input2.wav"))
# print(speech_to_text("/content/input2.wav"))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
token = os.getenv("HF_TOKEN")

def load_model(model_name):
    print(f"🚀 Loading model: {model_name}")

    try:
        # ✅ Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token
        )

        # ✅ Load model with memory optimization
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,   # 🔥 important
            token=token
        )

        # ✅ Move to device
        model = model.to(DEVICE)

        # ✅ Set evaluation mode
        model.eval()

        # ✅ Optional: half precision (ONLY if GPU)
        if DEVICE == "cuda":
            model = model.half()

        print("✅ Model ready for inference")

        return tokenizer, model

    except Exception as e:
        print("❌ Model loading failed:", str(e))
        raise e


# Task-based loader
from functools import lru_cache

@lru_cache(maxsize=2)
def get_model(task):
    if task == "indic_en":
        # return load_model("ai4bharat/indictrans2-indic-en-dist-200M")
        return load_model("ai4bharat/indictrans2-indic-en-1B")

    elif task == "en-indic":
        return load_model("ai4bharat/indictrans2-en-indic-1B")
    
    elif task == "indic_indic":
        # return load_model("ai4bharat/indictrans2-indic-indic-dist-320M")
        return load_model("ai4bharat/indictrans2-indic-indic-1B")
    
    else:
        raise ValueError("Invalid task")

# Load processor once
from IndicTransToolkit import IndicProcessor
ip = IndicProcessor(inference=True)
# @lru_cache(maxsize=1)
# def get_processor():
#     return IndicProcessor(inference=True)


import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import os 

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True,token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
        token=token
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations
# from helper_fun import batch_translate
# from load_model import get_model, ip   # import your lazy loader

def translate_text(text, src_lang, tgt_lang):

    if src_lang == "eng_Latn" and tgt_lang != "eng_Latn":
        tokenizer, model = get_model("en_indic")

    elif src_lang != "eng_Latn" and tgt_lang == "eng_Latn":
        tokenizer, model = get_model("indic_en")

    else:
        tokenizer, model = get_model("indic_indic")

    output = batch_translate(
        [text],
        src_lang,
        tgt_lang,
        model,
        tokenizer,
        ip
    )

    return output[0]

# text = "ನನ್ನ ನೆಚ್ಚಿನ ಹವ್ಯಾಸ ಓದು ಮತ್ತು ಬರವಣಿಗೆ. ನಾನು ಪ್ರತಿದಿನವೂ ಕನಿಷ್ಠ ಒಂದು ಗಂಟೆ ಪುಸ್ತಕಗಳನ್ನು ಓದುತ್ತೇನೆ. ಪುಸ್ತಕಗಳು ಜ್ಞಾನದ ಬಂಡಾರವಾಗಿದ್ದು, ಅವು ನನಗೆ ಹೊಸ ವಿಷಯಗಳನ್ನು ಕಲಿಸುತ್ತವೆ. ಬರವಣಿಗೆಯು ನನ್ನ ಭಾವನೆಗಳನ್ನು ಮತ್ತು ಆಲೋಚನೆಗಳನ್ನು ವ್ಯಕ್ತಪಡಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ. ಒಳ್ಳೆಯ ಹವ್ಯಾಸಗಳು ಜೀವನಕ್ಕೆ ಸಂತೋಷ ಮತ್ತು ಉತ್ಸಾಹವನ್ನು ತರುತ್ತವೆ, ಹಾಗಾಗಿ ಓದುವುದು ಬಹಳ ಮುಖ್ಯ."
# translated_text = translate_text(text,'kan_Knda','eng_Latn')

# print("Translated Text:", translated_text)

#Text-to-Speech

from elevenlabs.client import ElevenLabs
import os

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)
# tts_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

def text_to_speech(text, output_file="output.mp3"):
    audio_stream = tts_client.text_to_speech.convert(
        voice_id="hpp4J3VqNfWAUOO0d1Us",
        model_id="eleven_multilingual_v2",
        text=text
    )

    with open(output_file, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    return output_file

LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "bn": "ben_Beng"
}

# from modules.speech_to_text import speech_to_text
# from modules.translator import translate_text
# from modules.text_speech import text_to_speech

def translation(audio, target_lang):
    """
    audio: input audio file path
    target_lang: e.g. 'kan_Knda'
    """

    # =========================
    # 1. Speech → Text
    # =========================
    text, lang_code = speech_to_text(audio)

    print("Input Text:", text)
    print("Detected Lang:", lang_code[:2])

    # Map language
    src_lang = LANG_MAP.get(lang_code[:2], "eng_Latn")

    print("Mapped Source Lang:", src_lang)

    # =========================
    # 2. Translate
    # =========================
    translated_text = translate_text(text, src_lang, target_lang)

    print("Translated Text:", translated_text)

    # =========================
    # 3. Text → Speech
    # =========================
    output_audio = text_to_speech(translated_text)

    return output_audio

# translation('/content/input2.wav','hin_Deva')
