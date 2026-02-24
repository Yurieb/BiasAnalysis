from transformers import pipeline
from textblob import TextBlob

# Load Transformer model one time
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)


# -----------------------------
# RoBERTa Transformer Sentiment
# -----------------------------
def get_ml_sentiment(text: str):
    """
    Analyse sentiment using RoBERTa.
    Returns: (label, confidence)
    """

    if text is None:
        return "neutral", 0.0

    text = text.strip()

    if len(text) < 10:
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


# -----------------------------
# TextBlob Sentiment
# -----------------------------
def get_textblob_sentiment(text: str):
    """
    Analyse sentiment using TextBlob.
    Returns: (label, polarity -1 to +1)
    """

    if not text:
        return "neutral", 0.0

    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)

    if polarity > 0.05:
        label = "positive"
    elif polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, polarity



# Normalise TextBlob to 0–100
def normalize_textblob(polarity: float):
    return round((polarity + 1) / 2 * 100, 2)


# Narrative Direction (user-facing, replaces Positive/Negative/Neutral)
def to_narrative_direction(internal_label: str, polarity: float) -> tuple[str, int]:
    """
    Map internal sentiment to user-facing Narrative Direction.
    Returns: (display_label, score -100 to +100)
    -100 = strongly critical, 0 = balanced, +100 = strongly supportive
    """
    score = int(round(polarity * 100))

    if internal_label == "positive":
        if abs(polarity) >= 0.5:
            return "Strongly Supportive", score
        return "Leans Supportive", score
    elif internal_label == "negative":
        if abs(polarity) >= 0.5:
            return "Strongly Critical", score
        return "Leans Critical", score
    else:
        return "Balanced", score



# Dual Sentiment and Narrative Framing Score
def run_dual_sentiment(text: str):

    rob_label, rob_score = get_ml_sentiment(text)
    roberta_percent = round(rob_score * 100, 2)

    tb_label, tb_score = get_textblob_sentiment(text)
    textblob_percent = normalize_textblob(tb_score)

    narrative_direction_label, narrative_direction_score = to_narrative_direction(
        rob_label, tb_score
    )

    # Framing Intensity 0-100 — how strongly the article pushes in that direction
    framing_intensity = int(round(roberta_percent))

    agreement = rob_label == tb_label

    return {
        "roberta_label": rob_label,
        "roberta_score": rob_score,
        "roberta_percent": roberta_percent,

        "textblob_label": tb_label,
        "textblob_score": tb_score,
        "textblob_percent": textblob_percent,

        "narrative_direction_label": narrative_direction_label,
        "narrative_direction_score": narrative_direction_score,
        "framing_intensity": framing_intensity,

        "agreement": agreement,
    }