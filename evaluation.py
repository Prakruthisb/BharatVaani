"""
BharatVaani Evaluation Script
Calculates WER (ASR) and BLEU (Translation) metrics
Uses streaming — no large downloads, no disk space issues.
Usage: python evaluation.py --lang hi --samples 100
"""

import argparse
import json
import os
import tempfile
from tqdm import tqdm

from jiwer import wer, cer
import sacrebleu
import soundfile as sf
from datasets import load_dataset
from src.pipeline import speech_to_text, translate_text
from dotenv import load_dotenv

load_dotenv()  # Load .env for API keys

# ── CONFIG ─────────────────────────────────────────────────────────────────────

LANGUAGE_CODES = {
    "hi": {"indicvoices": "Hindi",     "indicst": "hindi",     "flores": "hin_Deva", "display": "Hindi"},
    "ta": {"indicvoices": "Tamil",     "indicst": "tamil",     "flores": "tam_Taml", "display": "Tamil"},
    "te": {"indicvoices": "Telugu",    "indicst": "telugu",    "flores": "tel_Telu", "display": "Telugu"},
    "kn": {"indicvoices": "Kannada",   "indicst": "kannada",   "flores": "kan_Knda", "display": "Kannada"},
    "ml": {"indicvoices": "Malayalam", "indicst": "malayalam", "flores": "mal_Mlym", "display": "Malayalam"},
    "bn": {"indicvoices": "Bengali",   "indicst": "bengali",   "flores": "ben_Beng", "display": "Bengali"},
    "gu": {"indicvoices": "Gujarati",  "indicst": "gujarati",  "flores": "guj_Gujr", "display": "Gujarati"},
    "mr": {"indicvoices": "Marathi",   "indicst": "marathi",   "flores": "mar_Deva", "display": "Marathi"},
}

TARGET_LANG_FLORES = "eng_Latn"


# ── PIPELINE WRAPPERS ──────────────────────────────────────────────────────────

def transcribe(audio_array, sampling_rate) -> str:
    """Saves audio array to temp wav, calls Sarvam ASR."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    sf.write(temp_path, audio_array, sampling_rate)
    try:
        transcript, _ = speech_to_text(temp_path)
        return transcript.strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def translate(text, lang_code) -> str:
    """Calls IndicTrans2 translation."""
    src_lang = LANGUAGE_CODES[lang_code]["flores"]
    return translate_text(text, src_lang, TARGET_LANG_FLORES)


# ── DATA LOADING (STREAMING) ───────────────────────────────────────────────────

def load_asr_data_streaming(lang_code: str, n_samples: int):
    """
    Streams IndicVoices-R — no full download.
    Falls back to IndicVoices-ST if needed.
    """
    lang_name = LANGUAGE_CODES[lang_code]["indicvoices"]
    print(f"Streaming IndicVoices-R ({lang_name})...")
    try:
        ds = load_dataset(
            "ai4bharat/indicvoices_r",
            lang_name,
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        samples = list(ds.take(n_samples))
        print(f"  Fields: {list(samples[0].keys())}")
        return samples, "indicvoices_r"
    except Exception as e:
        print(f"  indicvoices_r failed: {e}")
        print("  Falling back to IndicVoices-ST...")
        lang_st = LANGUAGE_CODES[lang_code]["indicst"]
        ds = load_dataset(
            "ai4bharat/IndicVoices-ST",
            "indic2en",
            split=lang_st,
            streaming=True,
            trust_remote_code=True,
        )
        samples = list(ds.take(n_samples))
        print(f"  Fields: {list(samples[0].keys())}")
        return samples, "indicvoices_st"


def load_translation_refs_streaming(lang_code: str, n_samples: int):
    """Streams FLORES-200 devtest — no full download."""
    flores_code = LANGUAGE_CODES[lang_code]["flores"]
    pair = f"{flores_code}-{TARGET_LANG_FLORES}"
    print(f"Streaming FLORES-200 ({pair})...")
    try:
        ds = load_dataset("facebook/flores", pair, split="devtest", streaming=True)
        samples = list(ds.take(n_samples))
        print(f"  Fields: {list(samples[0].keys())}")
        return samples
    except Exception as e1:
        print(f"  facebook/flores failed: {e1}")
        try:
            ds = load_dataset("Muennighoff/flores200", pair, split="devtest", streaming=True)
            samples = list(ds.take(n_samples))
            print(f"  Fields: {list(samples[0].keys())}")
            return samples
        except Exception as e2:
            raise RuntimeError(f"Both FLORES sources failed.\n{e1}\n{e2}")


# ── EVALUATION ─────────────────────────────────────────────────────────────────

def evaluate_wer(lang_code: str, n_samples: int) -> dict:
    samples, source = load_asr_data_streaming(lang_code, n_samples)

    # Auto-detect transcript field
    sample0 = samples[0]
    if "normalized" in sample0:
        transcript_key = "normalized"
    elif "verbatim" in sample0:
        transcript_key = "verbatim"
    elif "transcript" in sample0:
        transcript_key = "transcript"
    elif "sentence" in sample0:
        transcript_key = "sentence"
    else:
        raise KeyError(f"No transcript field found. Available: {list(sample0.keys())}")

    print(f"  Using transcript field: '{transcript_key}'")

    references, hypotheses = [], []
    errors = 0

    for sample in tqdm(samples, desc="ASR inference"):
        try:
            audio    = sample["audio"]
            ref_text = sample[transcript_key].strip()
            hyp_text = transcribe(audio["array"], audio["sampling_rate"])
            references.append(ref_text)
            hypotheses.append(hyp_text)
        except Exception as e:
            errors += 1
            print(f"  ASR Error: {e}")

    if not references:
        return {"wer": None, "cer": None, "samples": 0, "errors": errors}

    return {
        "wer":     round(wer(references, hypotheses) * 100, 2),
        "cer":     round(cer(references, hypotheses) * 100, 2),
        "samples": len(references),
        "errors":  errors,
    }


def evaluate_bleu(lang_code: str, n_samples: int) -> dict:
    samples = load_translation_refs_streaming(lang_code, n_samples)

    # Auto-detect src/tgt field names
    sample0 = samples[0]
    flores_code = LANGUAGE_CODES[lang_code]["flores"]
    possible_src = [f"sentence_{flores_code}", "sentence_src", "sentence"]
    possible_tgt = [f"sentence_{TARGET_LANG_FLORES}", "sentence_tgt", "translation"]

    src_key = next((k for k in possible_src if k in sample0), None)
    tgt_key = next((k for k in possible_tgt if k in sample0), None)

    if not src_key or not tgt_key:
        raise KeyError(f"Could not find src/tgt fields. Available: {list(sample0.keys())}")

    print(f"  Using src: '{src_key}', tgt: '{tgt_key}'")

    hypotheses, references = [], []
    errors = 0

    for sample in tqdm(samples, desc="Translation inference"):
        try:
            src_text = sample[src_key].strip()
            ref_text = sample[tgt_key].strip()
            hyp_text = translate(src_text, lang_code).strip()
            hypotheses.append(hyp_text)
            references.append(ref_text)
        except Exception as e:
            errors += 1
            print(f"  Translation Error: {e}")

    if not hypotheses:
        return {"bleu": None, "samples": 0, "errors": errors}

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "bleu":    round(bleu.score, 2),
        "samples": len(hypotheses),
        "errors":  errors,
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BharatVaani Evaluation")
    parser.add_argument("--lang",    default="hi", choices=list(LANGUAGE_CODES.keys()))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--task",    default="both", choices=["wer", "bleu", "both"])
    parser.add_argument("--output",  default="eval_results.json")
    args = parser.parse_args()

    lang_display = LANGUAGE_CODES[args.lang]["display"]
    print(f"\n{'='*50}")
    print(f"  BharatVaani Evaluation — {lang_display}")
    print(f"  Samples : {args.samples} | Task: {args.task}")
    print(f"{'='*50}\n")

    results = {"language": lang_display, "lang_code": args.lang}

    if args.task in ("wer", "both"):
        print("── ASR Evaluation (WER) ──")
        wer_results = evaluate_wer(args.lang, args.samples)
        results["asr"] = wer_results
        print(f"  WER : {wer_results['wer']}%")
        print(f"  CER : {wer_results['cer']}%\n")

    if args.task in ("bleu", "both"):
        print("── Translation Evaluation (BLEU) ──")
        bleu_results = evaluate_bleu(args.lang, args.samples)
        results["translation"] = bleu_results
        print(f"  BLEU: {bleu_results['bleu']}\n")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {args.output}")

    print(f"\n── Resume-ready metrics ──")
    if "asr" in results and results["asr"]["wer"] is not None:
        print(f"  WER of {results['asr']['wer']}% on {lang_display} ({results['asr']['samples']} samples)")
    if "translation" in results and results["translation"]["bleu"] is not None:
        print(f"  BLEU score of {results['translation']['bleu']} on {lang_display}→English ({results['translation']['samples']} samples)")


if __name__ == "__main__":
    main()