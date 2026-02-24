import re

# Words used to evoke strong emotion in headlines
EMOTIVE_WORDS = {
    # Strong negative emotion
    "shocking", "outrage", "outraged", "crisis", "disaster",
    "chaos", "fury", "furious", "devastating", "explosive",
    "alarming", "dangerous", "threat", "threatening",

    # Conflict language
    "slam", "slammed", "attack", "blow", "clash", "battle",
    "fight", "war", "accuse", "accused",

    # Dramatic intensifiers
    "dramatic", "massive", "huge", "extreme", "radical",
    "critical", "severe", "urgent"
}

# Certainty language: strong assertions, exaggeration frequency
CERTAINTY_WORDS = {
    "always", "never", "everyone", "no one",
    "nothing", "everything", "completely",
    "entirely", "totally", "absolutely",
    "all", "none"
}


def analyse_bias_language(text: str):
    """
    Lightweight, explainable bias indicators based on language.
    Returns normalised metrics (per 1000 words) for fair comparison across articles.
    """

    if not text:
        return {
            "emotive_ratio": 0.0,
            "certainty_per_1000": 0.0,
            "certainty_ratio": 0.0,
            "bias_intensity_score": 0,
            "bias_level": "low",
            "total_words": 0,
        }

    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)

    if total_words == 0:
        return {
            "emotive_ratio": 0.0,
            "certainty_per_1000": 0.0,
            "certainty_ratio": 0.0,
            "bias_intensity_score": 0,
            "bias_level": "low",
            "total_words": 0,
        }

    emotive_count = sum(1 for w in words if w in EMOTIVE_WORDS)
    certainty_count = sum(1 for w in words if w in CERTAINTY_WORDS)

    emotive_ratio = emotive_count / total_words
    certainty_ratio = certainty_count / total_words

    # Normalised: certainty language per 1000 words
    certainty_per_1000 = round((certainty_count / total_words) * 1000, 1) if total_words else 0

    # Bias Intensity Score 0-100 combines emotive and certainty
    # Emotive: 0-5% of words -> 0-50 points; Certainty: 0-2% -> 0-50 points
    emotive_score = min(emotive_ratio * 1000, 50)  
    certainty_score = min(certainty_ratio * 2500, 50)  
    bias_intensity_score = min(100, round(emotive_score + certainty_score))

    # Clear definitions for Low / Moderate / High
    if bias_intensity_score <= 25:
        bias_level = "low"
    elif bias_intensity_score <= 50:
        bias_level = "moderate"
    else:
        bias_level = "high"

    return {
        "emotive_ratio": round(emotive_ratio, 4),
        "certainty_per_1000": certainty_per_1000,
        "certainty_ratio": round(certainty_ratio, 4),
        "bias_intensity_score": bias_intensity_score,
        "bias_level": bias_level,
        "total_words": total_words,
    }