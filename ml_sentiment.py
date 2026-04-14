import re
import json
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


# Load Transformer model one time at startup
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

vader = SentimentIntensityAnalyzer()

# Gemini — configured once at startup from .env
_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key:
    genai.configure(api_key=_gemini_key)
    _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    _gemini_model = None


# -------------------------------------------------------
# Phase 1 fix: chunk text instead of hard-truncating to
# 512 chars. RoBERTa is run on up to MAX_CHUNKS chunks,
# then results are combined via majority vote.
# -------------------------------------------------------
_CHUNK_CHARS = 2000   # ~300-400 words per chunk
_MAX_CHUNKS  = 3


def _split_chunks(text: str) -> list:
    """Split text into overlapping chunks for RoBERTa."""
    chunks = []
    step = _CHUNK_CHARS - 200   # 200-char overlap
    pos  = 0
    while pos < len(text) and len(chunks) < _MAX_CHUNKS:
        chunk = text[pos: pos + _CHUNK_CHARS].strip()
        if len(chunk) >= 10:
            chunks.append(chunk)
        pos += step
    return chunks or [text[:_CHUNK_CHARS]]


# ----------------------------------------
# RoBERTa Transformer  (main ML engine)
# ----------------------------------------
def get_ml_sentiment(text: str):
    """
    Analyse sentiment using RoBERTa across multiple chunks.
    Returns: (label, avg_confidence 0-1)
    """
    if not text or len(text.strip()) < 3:
        return "neutral", 0.0

    chunks = _split_chunks(text.strip())
    label_counts  = {"positive": 0, "negative": 0, "neutral": 0}
    confidences   = []

    for chunk in chunks:
        try:
            # RoBERTa's hard token limit is 512; slice characters here
            # (roughly 3-4 chars per token for English, so 512 chars < 512 tokens)
            result     = sentiment_pipeline(chunk[:512])[0]
            raw_label  = result.get("label", "").lower()
            confidence = round(result.get("score", 0.0), 3)

            if "positive" in raw_label:
                label = "positive"
            elif "negative" in raw_label:
                label = "negative"
            else:
                label = "neutral"

            label_counts[label] += 1
            confidences.append(confidence)
        except Exception:
            continue

    if not confidences:
        return "neutral", 0.0

    majority_label  = max(label_counts, key=label_counts.get)
    avg_confidence  = round(sum(confidences) / len(confidences), 3)
    return majority_label, avg_confidence


# ----------------------------------------
# VADER (rule-based and negation-aware)
# ----------------------------------------
def get_vader_sentiment(text: str):
    if not text:
        return "neutral", 0.0

    scores   = vader.polarity_scores(text)
    compound = round(scores["compound"], 3)

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, compound


# ----------------------------------------
# TextBlob (lexicon baseline)
# ----------------------------------------
TEXTBLOB_SCALE_FACTOR = 3.0

def get_textblob_sentiment(text: str):
    if not text:
        return "neutral", 0.0

    blob        = TextBlob(text)
    raw_polarity = blob.sentiment.polarity

    # Scale to make small news-text values meaningful, clamp to [-1, +1]
    scaled = max(min(raw_polarity * TEXTBLOB_SCALE_FACTOR, 1.0), -1.0)
    scaled = round(scaled, 3)

    if scaled > 0.05:
        label = "positive"
    elif scaled < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, scaled


# Normalisation helper
def normalize_to_percent(score_minus1_to_1):
    return round((score_minus1_to_1 + 1) / 2 * 100, 2)


# ----------------------------------------
# Phase 2: Gemini with JSON prompt + political lean
# ----------------------------------------
def get_gemini_sentiment(text: str):
    """
    LLM-based sentiment + political lean scoring.
    Returns: (label, score -1 to +1, lean: left|center|right|none)
    """
    if not text or not _gemini_model:
        return "neutral", 0.0, "none"

    prompt = (
        "You are a media bias analyst. Analyse this news article's narrative framing.\n"
        "Return ONLY valid JSON with these exact keys:\n"
        '{"score": <number from -1.0 to +1.0>, "lean": "<left|center|right|none>"}\n'
        "score: -1.0 = strongly critical/negative framing, +1.0 = strongly supportive/positive framing\n"
        "lean: political lean of the article's framing (left / center / right / none)\n"
        "Return only the JSON object, no other text.\n\n"
        f"{text[:1500]}"
    )

    try:
        response = _gemini_model.generate_content(prompt)
        raw      = response.text.strip()

        # Extract JSON from response (handles ```json ... ``` wrappers)
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not json_match:
            return "neutral", 0.0, "none"

        data  = json.loads(json_match.group())
        score = float(data.get("score", 0.0))
        lean  = str(data.get("lean", "none")).lower().strip()

        if lean not in ("left", "center", "right", "none"):
            lean = "none"

        score = round(max(min(score, 1.0), -1.0), 3)

    except Exception:
        return "neutral", 0.0, "none"

    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, score, lean


# ----------------------------------------
# Hybrid Narrative Fusion (all 4 engines)
# ----------------------------------------
def compute_hybrid_narrative(
    rob_label,
    rob_conf,
    vader_score,
    tb_score,
    gemini_score=0.0,
):
    """
    Weighted fusion: 40% RoBERTa + 25% VADER + 15% TextBlob + 20% Gemini.
    Gemini defaults to 0.0 (neutral) if API key not set.
    """
    if rob_label == "positive":
        rob_direction = 1
    elif rob_label == "negative":
        rob_direction = -1
    else:
        rob_direction = 0

    roberta_component = rob_direction * rob_conf

    combined = (
        0.40 * roberta_component +
        0.25 * vader_score +
        0.15 * tb_score +
        0.20 * gemini_score
    )

    combined    = max(min(combined, 1), -1)
    final_score = int(round(combined * 100))

    if final_score >= 40:
        label = "Strongly Supportive"
    elif final_score > 10:
        label = "Leans Supportive"
    elif final_score <= -40:
        label = "Strongly Critical"
    elif final_score < -10:
        label = "Leans Critical"
    else:
        label = "Balanced"

    return final_score, label


# ----------------------------------------
# Full Sentiment Pipeline
# ----------------------------------------
def run_sentiment_pipeline(text: str):
    """
    Main entry point used by news_demo.py.
    Returns a dictionary consumed by the templates.
    """
    # RoBERTa
    rob_label, rob_conf = get_ml_sentiment(text)
    roberta_percent = round(rob_conf * 100, 2)

    # VADER
    vader_label, vader_score = get_vader_sentiment(text)
    vader_percent = normalize_to_percent(vader_score)

    # TextBlob
    tb_label, tb_score = get_textblob_sentiment(text)
    textblob_percent = normalize_to_percent(tb_score)

    # Gemini (now returns lean as 3rd value)
    gemini_label, gemini_score, gemini_lean = get_gemini_sentiment(text)
    gemini_percent = normalize_to_percent(gemini_score)

    # Hybrid narrative
    narrative_score, narrative_label = compute_hybrid_narrative(
        rob_label, rob_conf, vader_score, tb_score, gemini_score,
    )

    # Agreement across all 4 engines
    labels    = [rob_label, vader_label, tb_label, gemini_label]
    agreement = len(set(labels)) == 1

    # Divergence — max percentage-point spread between any two engines
    differences = [
        abs(roberta_percent  - vader_percent),
        abs(roberta_percent  - textblob_percent),
        abs(roberta_percent  - gemini_percent),
        abs(vader_percent    - textblob_percent),
        abs(vader_percent    - gemini_percent),
        abs(textblob_percent - gemini_percent),
    ]
    model_difference = round(max(differences), 2)

    if model_difference < 10:
        divergence_level = "Low"
    elif model_difference < 25:
        divergence_level = "Moderate"
    else:
        divergence_level = "High"

    # Phase 2: confidence gate — when models are far apart the hybrid
    # average becomes noise, so flag it explicitly instead of blending.
    if model_difference > 40:
        narrative_label = "Uncertain \u2014 Models Disagree"

    framing_intensity = int(round(roberta_percent))

    return {
        # Individual engines
        "roberta_label":   rob_label,
        "roberta_percent": roberta_percent,

        "vader_label":   vader_label,
        "vader_percent": vader_percent,

        "textblob_label":   tb_label,
        "textblob_percent": textblob_percent,

        "gemini_label":   gemini_label,
        "gemini_percent": gemini_percent,
        "gemini_lean":    gemini_lean,       # NEW: political lean from Gemini

        # Hybrid result
        "narrative_direction_score": narrative_score,
        "narrative_direction_label": narrative_label,
        "framing_intensity":         framing_intensity,

        # Model comparison
        "agreement":        agreement,
        "model_difference": model_difference,
        "divergence_level": divergence_level,
    }
