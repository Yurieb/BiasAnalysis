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

# Words that suggest exaggeration or absolute claims
ABSOLUTIST_WORDS = {
    "always", "never", "everyone", "no one",
    "nothing", "everything", "completely",
    "entirely", "totally", "absolutely",
    "all", "none"
}


def analyse_bias_language(text: str):
    """
    Lightweight, explainable bias indicators based on language.
    """

     # Handle empty text safely
    if not text:
        return {
            "emotive_ratio": 0.0,
            "absolutist_count": 0,
            "bias_level": "low"
        }

    # Tokenise text into lowercase words 
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)

    if total_words == 0:
        return {
            "emotive_ratio": 0.0,
            "absolutist_count": 0,
            "bias_level": "low"
        }

    # Count emotive and absolutist terms
    emotive_count = sum(1 for w in words if w in EMOTIVE_WORDS)
    absolutist_count = sum(1 for w in words if w in ABSOLUTIST_WORDS)

    # Calculate Emotional Ratio
    emotive_ratio = emotive_count / total_words

    # Simple threshold based bias classification
    if emotive_ratio > 0.01 or absolutist_count >= 2:
        bias_level = "moderate"
    else:
        bias_level = "low"

    return {
        "emotive_ratio": round(emotive_ratio, 4),
        "absolutist_count": absolutist_count,
        "bias_level": bias_level
    }