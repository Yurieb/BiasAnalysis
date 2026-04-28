"""
Tests for Flask routes — checks pages load correctly without crashing.
Does NOT scrape URLs or call any AI models.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from news_demo import app, db


@pytest.fixture
def client():
    """Create a test client with a fresh in-memory database."""
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            # Run migrations so all new columns exist in the test DB
            from news_demo import run_migrations
            run_migrations()
        yield client



# Homepage
def test_homepage_loads(client):
    response = client.get("/")
    assert response.status_code == 200

def test_homepage_contains_medialens(client):
    response = client.get("/")
    assert b"MediaLens" in response.data


# Compare page
def test_compare_page_loads(client):
    response = client.get("/compare")
    assert response.status_code == 200

def test_compare_page_has_form(client):
    response = client.get("/compare")
    assert b"form" in response.data.lower()


# History page
def test_history_page_loads(client):
    response = client.get("/history")
    assert response.status_code == 200


# Stats page
def test_stats_page_loads(client):
    response = client.get("/stats")
    assert response.status_code == 200

def test_stats_page_shows_zero_analyses(client):
    response = client.get("/stats")
    assert response.status_code == 200



# Feedback route
def test_feedback_rejects_missing_data(client):
    response = client.post("/feedback", data={})
    # Should return an error, not a 200
    assert response.status_code in [400, 422, 500]

def test_feedback_rejects_invalid_rating(client):
    response = client.post("/feedback", data={
        "article_id": "1",
        "rating": "999",
        "lean": "none"
    })
    assert response.status_code in [400, 422]


# CSV export
def test_export_csv_loads(client):
    response = client.post("/export-csv")
    # Should return CSV or empty response not a crash
    assert response.status_code == 200


def test_unknown_route_returns_404(client):
    response = client.get("/this-page-does-not-exist")
    assert response.status_code == 404
