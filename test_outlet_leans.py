"""
Tests for outlet_leans.py — political lean database lookups.
"""
from outlet_leans import get_outlet_info


# --- Known outlets return correct lean ---

def test_bbc_is_center():
    info = get_outlet_info("bbc.com")
    assert info["lean"] == "center"

def test_foxnews_is_right():
    info = get_outlet_info("foxnews.com")
    assert info["lean"] == "right"

def test_nytimes_is_center_left():
    info = get_outlet_info("nytimes.com")
    assert info["lean"] == "center-left"

def test_reuters_is_center():
    info = get_outlet_info("reuters.com")
    assert info["lean"] == "center"

def test_breitbart_is_far_right():
    info = get_outlet_info("breitbart.com")
    assert info["lean"] == "far-right"

def test_guardian_is_center_left():
    info = get_outlet_info("theguardian.com")
    assert info["lean"] == "center-left"


# --- Factuality checks ---

def test_reuters_factuality_very_high():
    info = get_outlet_info("reuters.com")
    assert info["factuality"] == "very-high"

def test_breitbart_factuality_low():
    info = get_outlet_info("breitbart.com")
    assert info["factuality"] == "low"

def test_nytimes_factuality_high():
    info = get_outlet_info("nytimes.com")
    assert info["factuality"] == "high"


# --- known flag ---

def test_known_outlet_returns_true():
    info = get_outlet_info("bbc.com")
    assert info["known"] is True

def test_unknown_outlet_returns_false():
    info = get_outlet_info("randomnewssite123.com")
    assert info["known"] is False

def test_unknown_outlet_lean_is_unknown():
    info = get_outlet_info("randomnewssite123.com")
    assert info["lean"] == "unknown"


# --- Subdomain stripping ---

def test_www_prefix_stripped():
    info = get_outlet_info("www.bbc.com")
    assert info["known"] is True
    assert info["lean"] == "center"

def test_news_prefix_stripped():
    info = get_outlet_info("news.bbc.com")
    assert info["known"] is True

def test_m_prefix_stripped():
    info = get_outlet_info("m.foxnews.com")
    assert info["known"] is True


# --- lean_position is a valid number ---

def test_lean_position_is_number():
    info = get_outlet_info("bbc.com")
    assert isinstance(info["lean_position"], int)
    assert 0 <= info["lean_position"] <= 100

def test_far_right_position_is_95():
    info = get_outlet_info("breitbart.com")
    assert info["lean_position"] == 95

def test_center_position_is_50():
    info = get_outlet_info("reuters.com")
    assert info["lean_position"] == 50


# --- Labels are human-readable strings ---

def test_lean_label_is_string():
    info = get_outlet_info("bbc.com")
    assert isinstance(info["lean_label"], str)
    assert len(info["lean_label"]) > 0

def test_factuality_label_is_string():
    info = get_outlet_info("bbc.com")
    assert isinstance(info["factuality_label"], str)
    assert len(info["factuality_label"]) > 0

def test_center_lean_label():
    info = get_outlet_info("reuters.com")
    assert info["lean_label"] == "Center"

def test_right_lean_label():
    info = get_outlet_info("foxnews.com")
    assert info["lean_label"] == "Right"
