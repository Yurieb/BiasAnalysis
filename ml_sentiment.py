from transformers import pipeline

# Load sentiment model once (important for performance)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def get_ml_sentiment(text):
    """
    Uses a Transformer-based model to analyse sentiment.
    Returns: (label, score)
    """

    # Truncate long articles (model limit)
    text = text[:512]

    result = sentiment_pipeline(text)[0]

    raw_label = result["label"]
    score = round(result["score"], 3)

    # Map model labels to your system labels
    if raw_label == "LABEL_2":
        label = "positive"
    elif raw_label == "LABEL_0":
        label = "negative"
    else:
        label = "neutral"

    return label, score
