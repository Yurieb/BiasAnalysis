from transformers import pipeline

# Load sentiment model one time for performance
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def get_ml_sentiment(text: str):
    """
    Analyse sentiment using a Transformer model.
    Returns: (label, confidence)
    """

    # Srict input validation
    if text is None:
        return "neutral", 0.0

    text = text.strip()

    if len(text) < 10:
        return "neutral", 0.0

    # Truncate long input
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
