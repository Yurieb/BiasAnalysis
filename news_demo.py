import os
import csv
import io
from datetime import datetime
from urllib.parse import urlparse

from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from newspaper import Article as NewsArticle
import trafilatura

from ml_sentiment import run_sentiment_pipeline
from bias_analysis import analyse_bias_language
from outlet_leans import get_outlet_info

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# -------------------------------------------------------
# Category detection
# -------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Politics": [
        "election", "vote", "government", "president", "congress", "parliament",
        "minister", "policy", "politician", "democrat", "republican", "senate",
        "legislation", "budget", "political", "campaign", "candidate",
    ],
    "Health": [
        "covid", "vaccine", "hospital", "doctor", "medical", "health", "patient",
        "disease", "treatment", "virus", "pandemic", "diagnosis", "therapy",
        "mental health", "clinical", "pharmaceutical",
    ],
    "Environment": [
        "climate", "environment", "carbon", "emissions", "renewable", "solar",
        "wind", "pollution", "biodiversity", "conservation", "sustainability",
        "wildfire", "flood", "drought", "green", "eco",
    ],
    "Sports": [
        "game", "match", "team", "player", "championship", "league", "score",
        "football", "soccer", "basketball", "tennis", "olympics", "tournament",
        "coach", "season", "playoff",
    ],
}

def detect_category(title: str, text: str) -> str:
    combined = f"{(title or '')} {(text or '')}".lower()
    scores = {cat: sum(1 for kw in kws if kw in combined)
              for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "General"


def confidence_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


# -------------------------------------------------------
# Database models
# -------------------------------------------------------
class Article(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    url        = db.Column(db.String(500), unique=True, nullable=False)
    title      = db.Column(db.String(300))
    source     = db.Column(db.String(200))
    text       = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AnalysisResult(db.Model):
    """Stores every engine result so we never need to re-analyse for CSV export."""
    id             = db.Column(db.Integer, primary_key=True)
    article_id     = db.Column(db.Integer, db.ForeignKey("article.id"), nullable=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    # RoBERTa (primary)
    sentiment_label = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)

    # All engines — stored so export_csv never re-calls Gemini
    narrative_score  = db.Column(db.Integer,   nullable=True)
    narrative_label  = db.Column(db.String(50), nullable=True)

    vader_label      = db.Column(db.String(20),  nullable=True)
    vader_percent    = db.Column(db.Float,        nullable=True)

    textblob_label   = db.Column(db.String(20),  nullable=True)
    textblob_percent = db.Column(db.Float,        nullable=True)

    gemini_label     = db.Column(db.String(20),  nullable=True)
    gemini_percent   = db.Column(db.Float,        nullable=True)
    gemini_lean      = db.Column(db.String(20),  nullable=True)   

    # Bias metrics
    bias_level          = db.Column(db.String(20), nullable=True)
    bias_score          = db.Column(db.Integer,    nullable=True)
    emotive_ratio       = db.Column(db.Float,      nullable=True)
    certainty_per_1000  = db.Column(db.Float,      nullable=True)
    total_words         = db.Column(db.Integer,    nullable=True)

    # Model agreement
    model_agreement  = db.Column(db.Boolean, nullable=True)
    divergence_level = db.Column(db.String(20), nullable=True)
    divergence_pct   = db.Column(db.Float,      nullable=True)

    # Category
    category = db.Column(db.String(50), nullable=True)

    article = db.relationship("Article", backref=db.backref("analyses", lazy=True))


class UserFeedback(db.Model):
    """Phase 3: stores user correction ratings for each analysis."""
    id         = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey("article.id"), nullable=False)
    rating     = db.Column(db.Integer)       
    user_lean  = db.Column(db.String(20))    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    article = db.relationship("Article", backref=db.backref("feedback", lazy=True))


# -------------------------------------------------------
# DB migration adds new columns to existing
# tables without destroying data already in news.db
# -------------------------------------------------------
def run_migrations():
    new_columns = [
        ("analysis_result", "narrative_score",  "INTEGER  DEFAULT 0"),
        ("analysis_result", "narrative_label",  "VARCHAR(50) DEFAULT 'Balanced'"),
        ("analysis_result", "vader_label",       "VARCHAR(20) DEFAULT 'neutral'"),
        ("analysis_result", "vader_percent",     "FLOAT    DEFAULT 50.0"),
        ("analysis_result", "textblob_label",    "VARCHAR(20) DEFAULT 'neutral'"),
        ("analysis_result", "textblob_percent",  "FLOAT    DEFAULT 50.0"),
        ("analysis_result", "gemini_label",      "VARCHAR(20) DEFAULT 'neutral'"),
        ("analysis_result", "gemini_percent",    "FLOAT    DEFAULT 50.0"),
        ("analysis_result", "gemini_lean",       "VARCHAR(20) DEFAULT 'none'"),
        ("analysis_result", "bias_level",        "VARCHAR(20) DEFAULT 'low'"),
        ("analysis_result", "bias_score",        "INTEGER  DEFAULT 0"),
        ("analysis_result", "emotive_ratio",     "FLOAT    DEFAULT 0.0"),
        ("analysis_result", "certainty_per_1000","FLOAT    DEFAULT 0.0"),
        ("analysis_result", "total_words",       "INTEGER  DEFAULT 0"),
        ("analysis_result", "model_agreement",   "BOOLEAN  DEFAULT 0"),
        ("analysis_result", "divergence_level",  "VARCHAR(20) DEFAULT 'Low'"),
        ("analysis_result", "divergence_pct",    "FLOAT    DEFAULT 0.0"),
        ("analysis_result", "category",          "VARCHAR(50) DEFAULT 'General'"),
    ]
    with db.engine.connect() as conn:
        for table, col, col_type in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass   


# -------------------------------------------------------
# Rescrape sources.txt only when the file changes on disk.
# -------------------------------------------------------
_analysis_cache: dict = {"results": None, "mtime": 0.0}

def run_sentiment_analysis(force: bool = False):
    global _analysis_cache

    sources_path = "sources.txt"
    try:
        mtime = os.path.getmtime(sources_path)
    except FileNotFoundError:
        return []

    if (not force
            and _analysis_cache["results"] is not None
            and mtime == _analysis_cache["mtime"]):
        return _analysis_cache["results"]

    urls = []
    with open(sources_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                line = line.split("|", 1)[1].strip()
            urls.append(line)

    results = []
    for url in urls:
        try:
            results.append(analyze_single_url(url))
        except Exception:
            continue

    _analysis_cache = {"results": results, "mtime": mtime}
    return results


# -------------------------------------------------------
# Core analysis function
# -------------------------------------------------------
def analyze_single_url(url: str) -> dict:
    a = NewsArticle(url)
    a.download()
    a.parse()

    title = a.title or ""
    body  = a.text  or ""

    # Fallback: trafilatura handles JS-heavy / paywalled sites
    if len(body.split()) < 50:
        downloaded = trafilatura.fetch_url(url)
        fallback   = trafilatura.extract(downloaded) or ""
        if len(fallback.split()) > len(body.split()):
            body = fallback

    if not body:
        body = title

    # Run all 4 engines
    sentiment_data = run_sentiment_pipeline(body)

    sentiment_label  = sentiment_data["roberta_label"]
    sentiment_score  = sentiment_data["roberta_percent"] / 100
    roberta_percent  = sentiment_data["roberta_percent"]

    vader_label      = sentiment_data["vader_label"]
    vader_percent    = sentiment_data["vader_percent"]

    textblob_label   = sentiment_data["textblob_label"]
    textblob_percent = sentiment_data["textblob_percent"]

    gemini_label     = sentiment_data.get("gemini_label",   "neutral")
    gemini_percent   = sentiment_data.get("gemini_percent", 50.0)
    gemini_lean      = sentiment_data.get("gemini_lean",    "none")    

    narrative_direction_label = sentiment_data["narrative_direction_label"]
    narrative_direction_score = sentiment_data["narrative_direction_score"]
    framing_intensity         = sentiment_data["framing_intensity"]

    agreement        = sentiment_data["agreement"]
    model_difference = sentiment_data["model_difference"]
    divergence_level = sentiment_data["divergence_level"]

    conf_level = confidence_level(sentiment_score)

    bias_info   = analyse_bias_language(body)
    category    = detect_category(title, body)
    source_domain = urlparse(url).netloc
    outlet_info = get_outlet_info(source_domain)

    article_row = Article.query.filter_by(url=url).first()
    if not article_row:
        article_row = Article(url=url, title=title, source=source_domain, text=body)
        db.session.add(article_row)
        db.session.commit()

    # Save full analysis so export_csv can read from DB
    analysis = AnalysisResult(
        article_id=article_row.id,

        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,

        narrative_score=narrative_direction_score,
        narrative_label=narrative_direction_label,

        vader_label=vader_label,
        vader_percent=vader_percent,

        textblob_label=textblob_label,
        textblob_percent=textblob_percent,

        gemini_label=gemini_label,
        gemini_percent=gemini_percent,
        gemini_lean=gemini_lean,

        bias_level=bias_info["bias_level"],
        bias_score=bias_info["bias_intensity_score"],
        emotive_ratio=bias_info["emotive_ratio"],
        certainty_per_1000=bias_info["certainty_per_1000"],
        total_words=bias_info["total_words"],

        model_agreement=agreement,
        divergence_level=divergence_level,
        divergence_pct=model_difference,

        category=category,
    )
    db.session.add(analysis)
    db.session.commit()

    #  needed for feedback
    return {
        "article_id": article_row.id,       
        "title":    title,
        "source":   source_domain,
        "url":      url,
        "category": category,

        # Narrative framing
        "narrative_direction_label": narrative_direction_label,
        "narrative_direction_score": narrative_direction_score,
        "framing_intensity":         framing_intensity,

        # Individual engines
        "roberta_label":   sentiment_label,
        "roberta_percent": roberta_percent,
        "confidence_level": conf_level,

        "vader_label":   vader_label,
        "vader_percent": vader_percent,

        "textblob_label":   textblob_label,
        "textblob_percent": textblob_percent,

        "gemini_label":   gemini_label,
        "gemini_percent": gemini_percent,
        "gemini_lean":    gemini_lean,          

        # Legacy fields used by history page
        "sentiment":       sentiment_label,
        "sentiment_score": sentiment_score,

        # Model agreement / divergence
        "agreement":        agreement,
        "model_difference": model_difference,
        "divergence_level": divergence_level,

        "bias": bias_info,

        # Outlet political lean 
        "outlet_lean":          outlet_info["lean"],
        "outlet_lean_label":    outlet_info["lean_label"],
        "outlet_lean_position": outlet_info["lean_position"],
        "outlet_factuality":    outlet_info["factuality"],
        "outlet_factuality_label": outlet_info["factuality_label"],
        "outlet_known":         outlet_info["known"],
    }


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def index():
    analysis_results = run_sentiment_analysis()
    return render_template("results.html", analysis_results=analysis_results, has_new=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url", "").strip()
    if not url:
        return redirect(url_for("index"))

    new_result   = analyze_single_url(url)
    batch_results = run_sentiment_analysis()
    analysis_results = [new_result] + batch_results

    return render_template("results.html", analysis_results=analysis_results, has_new=True)


@app.route("/history")
def history():
    results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).all()
    return render_template("history.html", results=results)


# -------------------------------------------------------
# Source Comparison
# -------------------------------------------------------
def calculate_comparison(results: list) -> dict:
    scores = [r["narrative_direction_score"] for r in results]
    spread = max(scores) - min(scores)

    most_positive = max(results, key=lambda x: x["narrative_direction_score"])
    most_critical = min(results, key=lambda x: x["narrative_direction_score"])
    most_biased   = max(results, key=lambda x: x["bias"]["bias_intensity_score"])

    if spread >= 60:
        verdict = "Strong framing differences detected across sources"
    elif spread >= 30:
        verdict = "Moderate framing differences detected across sources"
    else:
        verdict = "Sources show broadly similar framing on this story"

    return {
        "spread":        spread,
        "average_score": round(sum(scores) / len(scores), 1),
        "most_positive": most_positive,
        "most_critical": most_critical,
        "most_biased":   most_biased,
        "verdict":       verdict,
    }


@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "POST":
        urls      = request.form.getlist("urls")
        labels    = request.form.getlist("labels")
        countries = request.form.getlist("countries")

        sources = [
            (url.strip(), label.strip(), country.strip())
            for url, label, country in zip(urls, labels, countries)
            if url.strip()
        ]

        if len(sources) < 2:
            return render_template(
                "compare_form.html",
                error="Please enter at least 2 URLs to compare."
            )

        results = []
        errors  = []
        for url, label, country in sources:
            try:
                result = analyze_single_url(url)
                result["outlet_label"]   = label or url
                result["outlet_country"] = country or "Unknown"
                results.append(result)
            except Exception:
                errors.append(f"Could not analyse: {url}")

        if len(results) < 2:
            return render_template(
                "compare_form.html",
                error="Could not scrape enough articles. Try different URLs."
            )

        comparison = calculate_comparison(results)
        return render_template(
            "compare_results.html",
            results=results,
            comparison=comparison,
            errors=errors,
        )

    return render_template("compare_form.html", error=None)


# -------------------------------------------------------
# export_csv now reads stored DB results
# -------------------------------------------------------
@app.route("/export-csv", methods=["POST"])
def export_csv():
    urls      = request.form.getlist("urls")
    labels    = request.form.getlist("labels")
    countries = request.form.getlist("countries")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Outlet", "Country", "Title", "Source", "URL",
        "Narrative Score", "Narrative Label",
        "RoBERTa", "VADER", "TextBlob", "Gemini", "Gemini Lean",
        "Model Agreement", "Divergence Level", "Divergence %",
        "Bias Level", "Bias Score",
        "Emotional %", "Certainty per 1k", "Article Words",
    ])

    for url, label, country in zip(urls, labels, countries):
        url = url.strip()
        if not url:
            continue

        # Look up stored result no re-analysis needed
        article = Article.query.filter_by(url=url).first()
        if not article:
            continue
        result = (
            AnalysisResult.query
            .filter_by(article_id=article.id)
            .order_by(AnalysisResult.created_at.desc())
            .first()
        )
        if not result:
            continue

        writer.writerow([
            label or article.source,
            country or "Unknown",
            article.title,
            article.source,
            url,
            result.narrative_score,
            result.narrative_label,
            result.sentiment_label,       # RoBERTa
            result.vader_label,
            result.textblob_label,
            result.gemini_label,
            result.gemini_lean or "none",
            "Yes" if result.model_agreement else "No",
            result.divergence_level,
            result.divergence_pct,
            result.bias_level,
            result.bias_score,
            round((result.emotive_ratio or 0) * 100, 1),
            result.certainty_per_1000,
            result.total_words,
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=medialens_comparison.csv"},
    )


# -------------------------------------------------------
# Correction ratings
# -------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def submit_feedback():
    article_id = request.form.get("article_id", type=int)
    rating     = request.form.get("rating",     type=int)
    user_lean  = request.form.get("lean",       "none").strip().lower()

    if not article_id or rating not in (1, -1):
        return jsonify({"error": "invalid"}), 400

    if user_lean not in ("left", "center", "right", "none"):
        user_lean = "none"

    feedback = UserFeedback(
        article_id=article_id,
        rating=rating,
        user_lean=user_lean,
    )
    db.session.add(feedback)
    db.session.commit()
    return jsonify({"ok": True}), 200


@app.route("/stats")
def stats():
    total_analyses = AnalysisResult.query.count()
    total_feedback = UserFeedback.query.count()
    positive_count = UserFeedback.query.filter_by(rating=1).count()
    negative_count = UserFeedback.query.filter_by(rating=-1).count()

    accuracy_pct = (
        round(positive_count / total_feedback * 100, 1)
        if total_feedback else None
    )

    # Lean breakdown from user corrections
    lean_counts = {
        lean: UserFeedback.query.filter_by(user_lean=lean).count()
        for lean in ("left", "center", "right", "none")
    }

    # Top sources by analysis count
    from sqlalchemy import func
    top_sources = (
        db.session.query(Article.source, func.count(AnalysisResult.id).label("cnt"))
        .join(AnalysisResult, Article.id == AnalysisResult.article_id)
        .group_by(Article.source)
        .order_by(func.count(AnalysisResult.id).desc())
        .limit(10)
        .all()
    )

    # Category breakdown
    category_counts = (
        db.session.query(AnalysisResult.category, func.count(AnalysisResult.id).label("cnt"))
        .filter(AnalysisResult.category.isnot(None))
        .group_by(AnalysisResult.category)
        .order_by(func.count(AnalysisResult.id).desc())
        .all()
    )

    # Engine divergence stats
    avg_divergence = db.session.query(
        func.avg(AnalysisResult.divergence_pct)
    ).scalar() or 0

    return render_template(
        "stats.html",
        total_analyses=total_analyses,
        total_feedback=total_feedback,
        positive_count=positive_count,
        negative_count=negative_count,
        accuracy_pct=accuracy_pct,
        lean_counts=lean_counts,
        top_sources=top_sources,
        category_counts=category_counts,
        avg_divergence=round(avg_divergence, 1),
    )


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        run_migrations()
    app.run(debug=True)
