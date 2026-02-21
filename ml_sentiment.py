from transformers import pipeline

# Load sentiment model one time for performance
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)
from transformers import pipeline
from textblob import TextBlob


# Load Transformer model one time
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)



# RoBERTa Transformer Sentiment (Main ML Model)
def get_ml_sentiment(text: str):
    """
    Analyse sentiment using RoBERTa Transformer model.
    Returns: (label, confidence)
    """

    # Strict input validation
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

# TextBlob Baseline Sentiment (Lexicon-Based Model)
def get_textblob_sentiment(text: str):
    """
    Analyse sentiment using TextBlob.
    This is a lexicon-based baseline model.
    Returns: (label, polarity score from -1 to +1)
    """

    if not text:
        return "neutral", 0.0

    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)

    # TextBlob polarity range: -1 to +1
    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return label, polarity

# Dual Sentiment Wrapper Comparison Layer
def run_dual_sentiment(text: str):
    """
    Runs BOTH:
    - RoBERTa (Transformer ML)
    - TextBlob (Lexicon baseline)

    Returns structured comparison.
    """

    # Run RoBERTa
    rob_label, rob_score = get_ml_sentiment(text)

    # Run TextBlob
    tb_label, tb_score = get_textblob_sentiment(text)

    # Check if both models agree
    agreement = rob_label == tb_label

    return {
        "roberta_label": rob_label,
        "roberta_score": rob_score,
        "textblob_label": tb_label,
        "textblob_score": tb_score,
        "agreement": agreement
    }
