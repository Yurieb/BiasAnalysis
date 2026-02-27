from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY 



# Load Transformer model one time
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

vader = SentimentIntensityAnalyzer()

# Gemini — configured once at startup from .env
_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key:
    genai.configure(api_key=_gemini_key)
    _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    _gemini_model = None


# ----------------------------------------
# RoBERTa Transformer  (main ML engine)
# ----------------------------------------
def get_ml_sentiment(text: str):
    """
    Analyse sentiment using RoBERTa.
    Returns: (label, confidence 0–1)
    """

    if not text:
        return "neutral", 0.0

    text = text.strip()

    if len(text) < 3:
        return "neutral", 0.0

    text = text[:512]

    try:
        result = sentiment_pipeline(text)[0]
    except Exception:
        return "neutral", 0.0

    raw_label = result.get("label", "").lower()
    confidence = round(result.get("score", 0.0), 3)

    if "positive" in raw_label:
        label = "positive"
    elif "negative" in raw_label:
        label = "negative"
    else:
        label = "neutral"

    return label, confidence


# ----------------------------------------
# VADER (rule-based and negation-aware)
# ----------------------------------------
def get_vader_sentiment(text: str):
    if not text:
        return "neutral", 0.0

    scores = vader.polarity_scores(text)
    compound = round(scores["compound"], 3)  # -1 to +1

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

    blob = TextBlob(text)
    raw_polarity = blob.sentiment.polarity

    # Scale up to make small news-text values meaningful,
    # then clamp back to [-1, +1]
    scaled_polarity = max(min(raw_polarity * TEXTBLOB_SCALE_FACTOR, 1.0), -1.0)
    scaled_polarity = round(scaled_polarity, 3)

    # Label thresholds on the scaled value
    if scaled_polarity > 0.05:
        label = "positive"
    elif scaled_polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, scaled_polarity


# Normalisation helpers

def normalize_to_percent(score_minus1_to_1):
    return round((score_minus1_to_1 + 1) / 2 * 100, 2)


# ----------------------------------------
# Gemini (LLM-based 4th engine)
# ----------------------------------------
def get_gemini_sentiment(text: str):
    """
    Asks Gemini to rate sentiment from -1.0 to +1.0.
    Falls back to neutral (0.0) if API key is missing or call fails.
    """
    if not text or not _gemini_model:
        return "neutral", 0.0

    prompt = (
        "Rate the sentiment of this news article on a scale from -1.0 to +1.0.\n"
        "-1.0 = strongly negative/critical, 0.0 = neutral, +1.0 = strongly positive/supportive.\n"
        "Reply with ONLY a single number, nothing else.\n\n"
        f"Article: {text[:1500]}"
    )

    try:
        response = _gemini_model.generate_content(prompt)
        score = float(response.text.strip())
        score = round(max(min(score, 1.0), -1.0), 3)  # clamp to [-1, +1]
    except Exception:
        return "neutral", 0.0

    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, score


# ----------------------------------------
# Hybrid Narrative Fusion ( all 3 engines together)
# ----------------------------------------
def compute_hybrid_narrative(
    rob_label,
    rob_conf,
    vader_score,
    tb_score,
    gemini_score=0.0,
):
    """
    Hybrid weighted fusion — 4 engines.
    40% RoBERTa, 25% VADER, 15% TextBlob (pre-scaled), 20% Gemini.
    Gemini defaults to 0.0 (neutral) if API key is not set.
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

    combined = max(min(combined, 1), -1)
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

    # Gemini
    gemini_label, gemini_score = get_gemini_sentiment(text)
    gemini_percent = normalize_to_percent(gemini_score)

    # Hybrid narrative
    narrative_score, narrative_label = compute_hybrid_narrative(
        rob_label,
        rob_conf,
        vader_score,
        tb_score,
        gemini_score,
    )

    # Agreement across all 4 engines
    labels = [rob_label, vader_label, tb_label, gemini_label]
    agreement = len(set(labels)) == 1

    # Divergence (max distance between any two engines)
    differences = [
        abs(roberta_percent - vader_percent),
        abs(roberta_percent - textblob_percent),
        abs(roberta_percent - gemini_percent),
        abs(vader_percent - textblob_percent),
        abs(vader_percent - gemini_percent),
        abs(textblob_percent - gemini_percent),
    ]
    model_difference = round(max(differences), 2)

    if model_difference < 10:
        divergence_level = "Low"
    elif model_difference < 25:
        divergence_level = "Moderate"
    else:
        divergence_level = "High"

    framing_intensity = int(round(roberta_percent))

    return {
        # Individual engines
        "roberta_label": rob_label,
        "roberta_percent": roberta_percent,

        "vader_label": vader_label,
        "vader_percent": vader_percent,

        "textblob_label": tb_label,
        "textblob_percent": textblob_percent,

        # Gemini
        "gemini_label": gemini_label,
        "gemini_percent": gemini_percent,

        # Hybrid result
        "narrative_direction_score": narrative_score,
        "narrative_direction_label": narrative_label,
        "framing_intensity": framing_intensity,

        # Comparison
        "agreement": agreement,
        "model_difference": model_difference,
        "divergence_level": divergence_level,
    }