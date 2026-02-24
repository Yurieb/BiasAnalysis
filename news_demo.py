from flask import Flask, render_template, request, redirect, url_for
from newspaper import Article as NewsArticle

from ml_sentiment import run_dual_sentiment
from bias_analysis import analyse_bias_language
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re

app = Flask(__name__)

# DB config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Category keywords for classification
CATEGORY_KEYWORDS = {
    "Politics": [
        "election", "vote", "government", "president", "congress", "parliament",
        "minister", "policy", "politician", "democrat", "republican", "senate",
        "legislation", "budget", "political", "campaign", "candidate"
    ],
    "Health": [
        "covid", "vaccine", "hospital", "doctor", "medical", "health", "patient",
        "disease", "treatment", "virus", "pandemic", "diagnosis", "therapy",
        "mental health", "clinical", "pharmaceutical"
    ],
    "Environment": [
        "climate", "environment", "carbon", "emissions", "renewable", "solar",
        "wind", "pollution", "biodiversity", "conservation", "sustainability",
        "wildfire", "flood", "drought", "green", "eco"
    ],
    "Sports": [
        "game", "match", "team", "player", "championship", "league", "score",
        "football", "soccer", "basketball", "tennis", "olympics", "tournament",
        "coach", "season", "playoff"
    ],
}


def detect_category(title: str, text: str) -> str:
    combined = f"{(title or '')} {(text or '')}".lower()
    words = set(re.findall(r"\b\w+\b", combined))
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in combined)
        scores[category] = matches

    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "General"


# Confidence helper
def confidence_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.55:
        return "medium"
    else:
        return "low"


# Database models
class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), unique=True, nullable=False)
    title = db.Column(db.String(300))
    source = db.Column(db.String(200))
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey("article.id"), nullable=False)
    sentiment_label = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    article = db.relationship("Article", backref=db.backref("analyses", lazy=True))


# Analyse single URL
def analyze_single_url(url):
    a = NewsArticle(url)
    a.download()
    a.parse()

    text = a.text or a.title or ""

    # Run BOTH sentiment models
    sentiment_data = run_dual_sentiment(text)

    sentiment_label = sentiment_data["roberta_label"]
    sentiment_score = sentiment_data["roberta_score"]
    roberta_percent = sentiment_data["roberta_percent"]

    textblob_label = sentiment_data["textblob_label"]
    textblob_score = sentiment_data["textblob_score"]
    textblob_percent = sentiment_data["textblob_percent"]

    narrative_direction_label = sentiment_data["narrative_direction_label"]
    narrative_direction_score = sentiment_data["narrative_direction_score"]
    framing_intensity = sentiment_data["framing_intensity"]

    agreement = sentiment_data["agreement"]
    conf_level = confidence_level(sentiment_score)

    # Language bias
    bias_info = analyse_bias_language(text)

    # Category context
    category = detect_category(a.title or "", text)

    source_domain = urlparse(url).netloc

    article_row = Article.query.filter_by(url=url).first()
    if not article_row:
        article_row = Article(
            url=url,
            title=a.title,
            source=source_domain,
            text=text,
        )
        db.session.add(article_row)
        db.session.commit()

    analysis = AnalysisResult(
        article_id=article_row.id,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
    )
    db.session.add(analysis)
    db.session.commit()

    return {
        "title": a.title,
        "source": source_domain,
        "url": url,
        "category": category,

        # Narrative framing (user-facing)
        "narrative_direction_label": narrative_direction_label,
        "narrative_direction_score": narrative_direction_score,
        "framing_intensity": framing_intensity,

        # Internal for DB
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "roberta_percent": roberta_percent,
        "confidence_level": conf_level,

        "textblob_label": textblob_label,
        "textblob_score": textblob_score,
        "textblob_percent": textblob_percent,

        "agreement": agreement,

        "bias": bias_info,
    }


# Batch analysis
def run_sentiment_analysis():
    results = []

    try:
        with open("sources.txt") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

    for url in urls:
        try:
            results.append(analyze_single_url(url))
        except Exception:
            continue

    return results


# Routes
@app.route("/")
def index():
    analysis_results = run_sentiment_analysis()
    return render_template(
        "results.html",
        analysis_results=analysis_results,
        has_new=False
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url")
    if not url:
        return redirect(url_for("index"))

    new_result = analyze_single_url(url)
    batch_results = run_sentiment_analysis()
    analysis_results = [new_result] + batch_results

    return render_template(
        "results.html",
        analysis_results=analysis_results,
        has_new=True
    )


@app.route("/history")
def history():
    results = AnalysisResult.query.order_by(
        AnalysisResult.created_at.desc()
    ).all()
    return render_template("history.html", results=results)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)