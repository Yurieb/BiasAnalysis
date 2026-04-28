"""
Outlet political lean and factuality database.

Lean scale (7 points):
  far-left | left | center-left | center | center-right | right | far-right

Factuality (from Media Bias/Fact Check):
  very-high | high | mixed | low | very-low

Sources: AllSides.com, MediaBiasFactCheck.com
"""

OUTLET_DATA = {
    # -------------------------------------------------------
    # United States
    # -------------------------------------------------------
    "foxnews.com":           {"lean": "right",        "factuality": "mixed",     "country": "US"},
    "fox.com":               {"lean": "right",        "factuality": "mixed",     "country": "US"},
    "foxbusiness.com":       {"lean": "center-right", "factuality": "mixed",     "country": "US"},
    "foxsports.com":         {"lean": "center",       "factuality": "high",      "country": "US"},
    "msnbc.com":             {"lean": "left",         "factuality": "mixed",     "country": "US"},
    "cnn.com":               {"lean": "center-left",  "factuality": "mixed",     "country": "US"},
    "nytimes.com":           {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "washingtonpost.com":    {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "wsj.com":               {"lean": "center-right", "factuality": "high",      "country": "US"},
    "reuters.com":           {"lean": "center",       "factuality": "very-high", "country": "US"},
    "apnews.com":            {"lean": "center",       "factuality": "very-high", "country": "US"},
    "npr.org":               {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "pbs.org":               {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "breitbart.com":         {"lean": "far-right",    "factuality": "low",       "country": "US"},
    "huffpost.com":          {"lean": "left",         "factuality": "mixed",     "country": "US"},
    "thehill.com":           {"lean": "center",       "factuality": "high",      "country": "US"},
    "politico.com":          {"lean": "center",       "factuality": "high",      "country": "US"},
    "axios.com":             {"lean": "center",       "factuality": "high",      "country": "US"},
    "usatoday.com":          {"lean": "center",       "factuality": "high",      "country": "US"},
    "nbcnews.com":           {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "abcnews.go.com":        {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "cbsnews.com":           {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "vox.com":               {"lean": "left",         "factuality": "high",      "country": "US"},
    "slate.com":             {"lean": "left",         "factuality": "high",      "country": "US"},
    "nationalreview.com":    {"lean": "right",        "factuality": "mixed",     "country": "US"},
    "theatlantic.com":       {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "newyorker.com":         {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "motherjones.com":       {"lean": "left",         "factuality": "high",      "country": "US"},
    "nypost.com":            {"lean": "right",        "factuality": "mixed",     "country": "US"},
    "newsweek.com":          {"lean": "center",       "factuality": "mixed",     "country": "US"},
    "time.com":              {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "forbes.com":            {"lean": "center-right", "factuality": "high",      "country": "US"},
    "businessinsider.com":   {"lean": "center-left",  "factuality": "mixed",     "country": "US"},
    "theintercept.com":      {"lean": "left",         "factuality": "mixed",     "country": "US"},
    "jacobin.com":           {"lean": "far-left",     "factuality": "mixed",     "country": "US"},
    "dailywire.com":         {"lean": "right",        "factuality": "mixed",     "country": "US"},
    "thefederalist.com":     {"lean": "right",        "factuality": "low",       "country": "US"},
    "thedailybeast.com":     {"lean": "left",         "factuality": "mixed",     "country": "US"},
    "cnbc.com":              {"lean": "center",       "factuality": "high",      "country": "US"},
    "bloomberg.com":         {"lean": "center",       "factuality": "high",      "country": "US"},
    "marketwatch.com":       {"lean": "center",       "factuality": "high",      "country": "US"},
    "reason.com":            {"lean": "center-right", "factuality": "high",      "country": "US"},
    "newsmax.com":           {"lean": "right",        "factuality": "mixed",     "country": "US"},
    "oann.com":              {"lean": "far-right",    "factuality": "low",       "country": "US"},
    "infowars.com":          {"lean": "far-right",    "factuality": "very-low",  "country": "US"},
    "propublica.org":        {"lean": "center-left",  "factuality": "very-high", "country": "US"},
    "commondreams.org":      {"lean": "far-left",     "factuality": "mixed",     "country": "US"},

    # -------------------------------------------------------
    # United Kingdom
    # -------------------------------------------------------
    "bbc.com":               {"lean": "center",       "factuality": "high",      "country": "UK"},
    "bbc.co.uk":             {"lean": "center",       "factuality": "high",      "country": "UK"},
    "theguardian.com":       {"lean": "center-left",  "factuality": "high",      "country": "UK"},
    "thetimes.co.uk":        {"lean": "center-right", "factuality": "high",      "country": "UK"},
    "telegraph.co.uk":       {"lean": "right",        "factuality": "high",      "country": "UK"},
    "independent.co.uk":     {"lean": "center-left",  "factuality": "high",      "country": "UK"},
    "mirror.co.uk":          {"lean": "left",         "factuality": "mixed",     "country": "UK"},
    "dailymail.co.uk":       {"lean": "right",        "factuality": "mixed",     "country": "UK"},
    "express.co.uk":         {"lean": "right",        "factuality": "mixed",     "country": "UK"},
    "thesun.co.uk":          {"lean": "right",        "factuality": "mixed",     "country": "UK"},
    "ft.com":                {"lean": "center",       "factuality": "very-high", "country": "UK"},
    "economist.com":         {"lean": "center",       "factuality": "very-high", "country": "UK"},
    "spectator.co.uk":       {"lean": "right",        "factuality": "mixed",     "country": "UK"},
    "newstatesman.com":      {"lean": "left",         "factuality": "high",      "country": "UK"},
    "metro.co.uk":           {"lean": "center",       "factuality": "mixed",     "country": "UK"},

    # -------------------------------------------------------
    # Middle East
    # -------------------------------------------------------
    "aljazeera.com":         {"lean": "center-left",  "factuality": "mixed",     "country": "Qatar"},
    "timesofisrael.com":     {"lean": "center-right", "factuality": "high",      "country": "Israel"},
    "haaretz.com":           {"lean": "center-left",  "factuality": "high",      "country": "Israel"},
    "jpost.com":             {"lean": "center-right", "factuality": "mixed",     "country": "Israel"},
    "arabnews.com":          {"lean": "center-right", "factuality": "mixed",     "country": "Saudi Arabia"},
    "middleeasteye.net":     {"lean": "center-left",  "factuality": "mixed",     "country": "UK/ME"},
    "presstv.ir":            {"lean": "far-left",     "factuality": "low",       "country": "Iran"},
    "trtworld.com":          {"lean": "center-right", "factuality": "mixed",     "country": "Turkey"},
    "dawn.com":              {"lean": "center",       "factuality": "high",      "country": "Pakistan"},

    # -------------------------------------------------------
    # Russia / State Media
    # -------------------------------------------------------
    "rt.com":                {"lean": "far-right",    "factuality": "low",       "country": "Russia"},
    "tass.com":              {"lean": "far-right",    "factuality": "low",       "country": "Russia"},
    "sputniknews.com":       {"lean": "far-right",    "factuality": "low",       "country": "Russia"},

    # -------------------------------------------------------
    # China / State Media
    # -------------------------------------------------------
    "xinhuanet.com":         {"lean": "far-right",    "factuality": "low",       "country": "China"},
    "globaltimes.cn":        {"lean": "far-right",    "factuality": "low",       "country": "China"},
    "cgtn.com":              {"lean": "far-right",    "factuality": "low",       "country": "China"},
    "chinadaily.com.cn":     {"lean": "far-right",    "factuality": "low",       "country": "China"},

    # -------------------------------------------------------
    # Europe
    # -------------------------------------------------------
    "dw.com":                {"lean": "center",       "factuality": "high",      "country": "Germany"},
    "france24.com":          {"lean": "center",       "factuality": "high",      "country": "France"},
    "euronews.com":          {"lean": "center",       "factuality": "high",      "country": "EU"},
    "lemonde.fr":            {"lean": "center-left",  "factuality": "high",      "country": "France"},
    "spiegel.de":            {"lean": "center-left",  "factuality": "high",      "country": "Germany"},
    "politico.eu":           {"lean": "center",       "factuality": "high",      "country": "EU"},

    # -------------------------------------------------------
    # Canada
    # -------------------------------------------------------
    "cbc.ca":                {"lean": "center-left",  "factuality": "high",      "country": "Canada"},
    "globeandmail.com":      {"lean": "center-right", "factuality": "high",      "country": "Canada"},
    "nationalpost.com":      {"lean": "center-right", "factuality": "mixed",     "country": "Canada"},
    "torontostar.com":       {"lean": "center-left",  "factuality": "high",      "country": "Canada"},

    # -------------------------------------------------------
    # Australia
    # -------------------------------------------------------
    "abc.net.au":            {"lean": "center-left",  "factuality": "high",      "country": "Australia"},
    "smh.com.au":            {"lean": "center-left",  "factuality": "high",      "country": "Australia"},
    "theaustralian.com.au":  {"lean": "center-right", "factuality": "high",      "country": "Australia"},
    "news.com.au":           {"lean": "center-right", "factuality": "mixed",     "country": "Australia"},

    # -------------------------------------------------------
    # India
    # -------------------------------------------------------
    "ndtv.com":              {"lean": "center-left",  "factuality": "mixed",     "country": "India"},
    "thehindu.com":          {"lean": "center-left",  "factuality": "high",      "country": "India"},
    "hindustantimes.com":    {"lean": "center-right", "factuality": "mixed",     "country": "India"},
    "timesofindia.com":      {"lean": "center",       "factuality": "mixed",     "country": "India"},

    # -------------------------------------------------------
    # Health and Science specialist outlets
    # -------------------------------------------------------
    "webmd.com":             {"lean": "center",       "factuality": "high",      "country": "US"},
    "healthline.com":        {"lean": "center",       "factuality": "high",      "country": "US"},
    "medicalnewstoday.com":  {"lean": "center",       "factuality": "high",      "country": "UK"},
    "statnews.com":          {"lean": "center",       "factuality": "very-high", "country": "US"},
    "nejm.org":              {"lean": "center",       "factuality": "very-high", "country": "US"},
    "thelancet.com":         {"lean": "center",       "factuality": "very-high", "country": "UK"},
    "bmj.com":               {"lean": "center",       "factuality": "very-high", "country": "UK"},
    "sciencedaily.com":      {"lean": "center",       "factuality": "high",      "country": "US"},
    "livescience.com":       {"lean": "center",       "factuality": "high",      "country": "US"},
    "newscientist.com":      {"lean": "center-left",  "factuality": "high",      "country": "UK"},
    "nature.com":            {"lean": "center",       "factuality": "very-high", "country": "UK"},
    "scientificamerican.com":{"lean": "center-left",  "factuality": "very-high", "country": "US"},
    "mayoclinic.org":        {"lean": "center",       "factuality": "very-high", "country": "US"},
    "who.int":               {"lean": "center",       "factuality": "very-high", "country": "International"},
    "nih.gov":               {"lean": "center",       "factuality": "very-high", "country": "US"},
    "cdc.gov":               {"lean": "center",       "factuality": "very-high", "country": "US"},

    # -------------------------------------------------------
    # Sports (specialist outlets)
    # -------------------------------------------------------
    "espn.com":              {"lean": "center",       "factuality": "high",      "country": "US"},
    "skysports.com":         {"lean": "center",       "factuality": "high",      "country": "UK"},
    "talksport.com":         {"lean": "center-right", "factuality": "mixed",     "country": "UK"},
    "dailystar.co.uk":       {"lean": "right",        "factuality": "mixed",     "country": "UK"},
    "goal.com":              {"lean": "center",       "factuality": "high",      "country": "International"},
    "theathletic.com":       {"lean": "center",       "factuality": "high",      "country": "US"},
    "bleacherreport.com":    {"lean": "center",       "factuality": "mixed",     "country": "US"},
    "sportbible.com":        {"lean": "center",       "factuality": "mixed",     "country": "UK"},

    # -------------------------------------------------------
    # Technology
    # -------------------------------------------------------
    "theverge.com":          {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "wired.com":             {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "arstechnica.com":       {"lean": "center-left",  "factuality": "very-high", "country": "US"},
    "techcrunch.com":        {"lean": "center",       "factuality": "high",      "country": "US"},
    "engadget.com":          {"lean": "center-left",  "factuality": "high",      "country": "US"},
    "zdnet.com":             {"lean": "center",       "factuality": "high",      "country": "US"},
    "bbc.com/technology":    {"lean": "center",       "factuality": "high",      "country": "UK"},

    # -------------------------------------------------------
    # Ireland
    # -------------------------------------------------------
    "rte.ie":                {"lean": "center",       "factuality": "high",      "country": "Ireland"},
    "irishtimes.com":        {"lean": "center-left",  "factuality": "high",      "country": "Ireland"},
    "independent.ie":        {"lean": "center-right", "factuality": "mixed",     "country": "Ireland"},
    "thejournal.ie":         {"lean": "center-left",  "factuality": "high",      "country": "Ireland"},
    "irishexaminer.com":     {"lean": "center",       "factuality": "high",      "country": "Ireland"},
}

# Position on a 0-100 LeftRight spectrum bar
LEAN_POSITIONS = {
    "far-left":     5,
    "left":         18,
    "center-left":  33,
    "center":       50,
    "center-right": 67,
    "right":        82,
    "far-right":    95,
}

LEAN_LABELS = {
    "far-left":     "Far Left",
    "left":         "Left",
    "center-left":  "Center Left",
    "center":       "Center",
    "center-right": "Center Right",
    "right":        "Right",
    "far-right":    "Far Right",
}

FACTUALITY_LABELS = {
    "very-high": "Very High",
    "high":      "High",
    "mixed":     "Mixed",
    "low":       "Low",
    "very-low":  "Very Low",
}


def get_outlet_info(domain: str) -> dict:
    """
    Look up lean and factuality for a domain.
    Strips www/news/m subdomains before matching.
    Unknown outlets return lean='unknown'.
    """
    domain = domain.lower().strip()

    for prefix in ("www.", "news.", "m.", "mobile.", "amp.", "edition."):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
            break

    data = OUTLET_DATA.get(domain, {})

    lean       = data.get("lean",       "unknown")
    factuality = data.get("factuality", "unknown")
    country    = data.get("country",    "Unknown")

    return {
        "lean":          lean,
        "lean_label":    LEAN_LABELS.get(lean, "Unknown"),
        "lean_position": LEAN_POSITIONS.get(lean, 50),
        "factuality":    factuality,
        "factuality_label": FACTUALITY_LABELS.get(factuality, "Unknown"),
        "country":       country,
        "known":         lean != "unknown",
    }
