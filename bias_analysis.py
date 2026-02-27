import re

# Words used to evoke strong emotion in headlines
# Expanded from ~40 to ~150 words for better coverage across news topics
EMOTIVE_WORDS = {
    # Strong negative emotion
    "shocking", "outrage", "outraged", "crisis", "disaster",
    "chaos", "fury", "furious", "devastating", "explosive",
    "alarming", "dangerous", "threat", "threatening", "horrific",
    "terrible", "awful", "dreadful", "appalling", "horrifying",
    "terrifying", "nightmarish", "catastrophic", "tragic", "heartbreaking",
    "disturbing", "frightening", "dire", "grim", "harrowing",

    # Conflict & aggression language
    "slam", "slammed", "attack", "attacked", "blow", "clash", "clashes",
    "battle", "fight", "war", "accuse", "accused", "blasted", "hammered",
    "condemned", "denounced", "lashed", "slammed", "ripped", "tore",
    "crushed", "destroyed", "obliterated", "annihilated", "decimated",
    "ambush", "assault", "confrontation", "escalation", "standoff",
    "crackdown", "siege", "hostage", "violence", "brutal", "vicious",

    # Dramatic intensifiers
    "dramatic", "massive", "huge", "extreme", "radical",
    "critical", "severe", "urgent", "unprecedented", "extraordinary",
    "stunning", "bombshell", "explosive", "sensational", "staggering",
    "jaw-dropping", "mind-blowing", "unbelievable", "incredible", "astounding",
    "earth-shattering", "game-changing", "groundbreaking", "landmark",

    # Fear & panic language
    "panic", "fear", "scared", "frightened", "terror", "dread",
    "anxiety", "hysteria", "paranoia", "peril", "danger", "risk",
    "vulnerability", "exposed", "defenceless", "helpless", "desperate",

    # Outrage & moral language
    "shameful", "disgraceful", "scandalous", "corrupt", "betrayal",
    "betrayed", "lied", "deceived", "manipulated", "exploited",
    "abused", "victimised", "persecuted", "oppressed", "silenced",
    "suppressed", "censored", "banned", "blocked", "forbidden",

    # Loaded political language
    "regime", "puppet", "propaganda", "brainwashing", "indoctrination",
    "extremist", "radical", "fanatic", "militant", "insurgent",
    "coup", "tyranny", "tyrannical", "authoritarian", "dictator",

    # Positive hype language (also emotive — just positive direction)
    "triumphant", "glorious", "heroic", "magnificent", "spectacular",
    "phenomenal", "extraordinary", "outstanding", "brilliant", "genius",
    "revolutionary", "visionary", "historic", "legendary", "iconic"
}

# Certainty / absolutist language — overstates facts, sign of biased writing
CERTAINTY_WORDS = {
    # Classic absolutist words
    "always", "never", "everyone", "nobody", "no one",
    "nothing", "everything", "completely", "entirely",
    "totally", "absolutely", "utterly", "perfectly",
    "all", "none", "every", "any", "anywhere", "everywhere",
    "nowhere", "forever", "impossible", "inevitable", "certain",
    "definitely", "undoubtedly", "unquestionably", "obviously",
    "clearly", "plainly", "simply", "merely", "just",

    # Exaggerated frequency
    "constantly", "continuously", "endlessly", "repeatedly",
    "invariably", "without exception", "in every case",

    # False certainty in reporting
    "proves", "confirms", "demonstrates", "shows conclusively",
    "undeniable", "irrefutable", "incontrovertible", "beyond doubt"
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

def compute_bias_intensity(emotive_ratio, certainty_per_1000):

    # Convert emotional % (0–1) to 0–100 scale
    emotional_score = emotive_ratio * 100

    # Normalize certainty (cap at 10 per 1k words)
    certainty_score = min(certainty_per_1000, 10) * 10

    # Weighted bias intensity
    bias_score = int(round(
        (0.6 * emotional_score) +
        (0.4 * certainty_score)
    ))

    # Clamp
    bias_score = min(bias_score, 100)

    if bias_score < 25:
        level = "Low"
    elif bias_score < 50:
        level = "Moderate"
    else:
        level = "High"

    return bias_score, level