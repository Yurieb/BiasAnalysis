"""
Tests for sentiment engines and bias analysis.
Does NOT call Gemini or RoBERTa (no API key / GPU needed).
"""
from ml_sentiment import get_vader_sentiment, get_textblob_sentiment
from bias_analysis import analyse_bias_language


# -------------------------------------------------------
# VADER tests
# get_vader_sentiment returns a tuple: (label, score)
# -------------------------------------------------------

def test_vader_positive_sentence():
    label, score = get_vader_sentiment("This is a wonderful and amazing achievement for everyone.")
    assert label == "positive"

def test_vader_negative_sentence():
    label, score = get_vader_sentiment("This is a terrible disaster and a catastrophic failure.")
    assert label == "negative"

def test_vader_neutral_sentence():
    label, score = get_vader_sentiment("The meeting was held on Tuesday at the office.")
    assert label == "neutral"

def test_vader_returns_score():
    label, score = get_vader_sentiment("Great news today!")
    assert -1.0 <= score <= 1.0

def test_vader_returns_label():
    label, score = get_vader_sentiment("Some text here.")
    assert label in ["positive", "negative", "neutral"]


# -------------------------------------------------------
# TextBlob tests
# get_textblob_sentiment returns a tuple: (label, score)
# -------------------------------------------------------

def test_textblob_positive_sentence():
    label, score = get_textblob_sentiment("This is an excellent and fantastic outcome.")
    assert label == "positive"

def test_textblob_negative_sentence():
    label, score = get_textblob_sentiment("This is a horrible and dreadful situation.")
    assert label == "negative"

def test_textblob_returns_label():
    label, score = get_textblob_sentiment("The report was published yesterday.")
    assert label in ["positive", "negative", "neutral"]

def test_textblob_returns_score():
    label, score = get_textblob_sentiment("Good results were announced.")
    assert isinstance(score, float)


# -------------------------------------------------------
# Bias analysis tests
# -------------------------------------------------------

def test_bias_low_on_neutral_text():
    text = "The government held a meeting to discuss the new policy proposal on Tuesday."
    result = analyse_bias_language(text)
    assert result["bias_level"] == "low"

def test_bias_high_on_dramatic_text():
    text = " ".join([
        "shocking outrage crisis disaster chaos fury devastating explosive alarming",
        "catastrophic horrific terrible awful appalling nightmarish always never",
        "absolutely certainly completely definitely undeniably unquestionably",
    ] * 5)
    result = analyse_bias_language(text)
    assert result["bias_level"] in ["moderate", "high"]

def test_bias_returns_emotive_ratio():
    text = "The president announced a new policy."
    result = analyse_bias_language(text)
    assert "emotive_ratio" in result
    assert 0.0 <= result["emotive_ratio"] <= 1.0

def test_bias_returns_certainty_per_1000():
    text = "Scientists always say the climate is definitely changing and it will never stop."
    result = analyse_bias_language(text)
    assert "certainty_per_1000" in result
    assert result["certainty_per_1000"] >= 0

def test_bias_returns_total_words():
    text = "This is a five word sentence plus more words here."
    result = analyse_bias_language(text)
    assert "total_words" in result
    assert result["total_words"] > 0

def test_bias_returns_bias_intensity_score():
    text = "The economy grew steadily last quarter according to official figures."
    result = analyse_bias_language(text)
    assert "bias_intensity_score" in result
    assert 0 <= result["bias_intensity_score"] <= 100

def test_bias_level_values_are_valid():
    text = "Some article text here about recent events."
    result = analyse_bias_language(text)
    assert result["bias_level"] in ["low", "moderate", "high"]

def test_empty_text_does_not_crash():
    result = analyse_bias_language("")
    assert "bias_level" in result

def test_long_text_does_not_crash():
    text = "The government announced a new policy. " * 200
    result = analyse_bias_language(text)
    assert "bias_level" in result
