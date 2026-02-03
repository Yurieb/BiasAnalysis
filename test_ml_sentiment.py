from ml_sentiment import get_ml_sentiment


def test_returns_valid_label():
    label, conf = get_ml_sentiment(
        "This is a very positive and successful outcome"
    )
    assert label in {"positive", "neutral", "negative"}
    assert 0.0 <= conf <= 1.0


def test_negative_language_not_positive():
    label, conf = get_ml_sentiment(
        "This decision is a complete failure and disaster"
    )
    assert label != "positive"


def test_empty_text():
    label, conf = get_ml_sentiment("")
    assert label == "neutral"
    assert conf == 0.0


def test_short_text():
    label, conf = get_ml_sentiment("OK")
    assert label == "neutral"
    assert conf == 0.0
