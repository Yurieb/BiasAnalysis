from flask import Flask, render_template, request, redirect, url_for
from newspaper import Article as NewsArticle
from ml_sentiment import get_ml_sentiment
from bias_analysis import analyse_bias_language
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# DB config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# DATABASE MODELS
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
    sentiment_score = db.Column(db.Float)  # confidence
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    article = db.relationship("Article", backref=db.backref("analyses", lazy=True))

# Analyse single URL
def analyze_single_url(url):
    a = NewsArticle(url)
    a.download()
    a.parse()
    
    text = a.text or a.title or ""
    sentiment_label, sentiment_score = get_ml_sentiment(text)
    bias_info = analyse_bias_language(text)
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
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "bias": bias_info
    }


# Batch analysis
def run_sentiment_analysis():
    results = []

    try:
        with open("sources.txt") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return [{
            "title": "Error: sources.txt not found",
            "source": "N/A",
            "sentiment": "error",
            "sentiment_score": 0,
            "bias": {"bias_level": "low"}
        }]

    for url in urls:
        try:
            a = NewsArticle(url)
            a.download()
            a.parse()

            text = a.text or a.title or ""
            sentiment_label, sentiment_score = get_ml_sentiment(text)
            bias_info = analyse_bias_language(text)
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

            results.append({
                "title": a.title,
                "source": source_domain,
                "url": url,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "bias": bias_info
            })

        except Exception as e:
            results.append({
                "title": f"FAILED: {e}",
                "source": urlparse(url).netloc,
                "sentiment": "error",
                "sentiment_score": 0,
                "bias": {"bias_level": "low"}
            })

    return results


# ROUTES
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
