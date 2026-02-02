from flask import Flask, render_template, request, redirect, url_for
from newspaper import Article as NewsArticle
from ml_sentiment import get_ml_sentiment
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# DB config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Sentiment thresholds 
# Polarity > 0.20 is Positive
POS_THRESHOLD = 0.20
# Polarity < -0.20 is Negative
NEG_THRESHOLD = -0.20


# DATABASE MODELS
class Article(db.Model):
    # Base article data
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), unique=True, nullable=False)
    title = db.Column(db.String(300))
    source = db.Column(db.String(200))
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AnalysisResult(db.Model):
    # Sentiment analysis result
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey("article.id"), nullable=False)
    sentiment_label = db.Column(db.String(20))  # positive / neutral / negative
    sentiment_score = db.Column(db.Float)       # TextBlob polarity
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    article = db.relationship("Article", backref=db.backref("analyses", lazy=True))


# Sentiment helper
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > POS_THRESHOLD:
        label = "positive"
    elif polarity < NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return label, polarity



# Scrapes, analyzes, and persists a single URL
def analyze_single_url(url):
    """Analyse one URL and return dict for template."""
    a = NewsArticle(url)
    a.download()
    a.parse()

    text = a.text or a.title
    sentiment_label, sentiment_score = get_sentiment(text)
    source_domain = urlparse(url).netloc

    # store article in db
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

    # Save new AnalysisResult record
    analysis = AnalysisResult(
        article_id=article_row.id,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
    )
    db.session.add(analysis)
    db.session.commit()

    # Return display data
    return {
        "title": a.title,
        "source": source_domain,
        "url": url,
        "sentiment": sentiment_label,
        "sentiment_score": round(sentiment_score, 3),
    }


# Main analysis for urls from sources.txt
def run_sentiment_analysis():
    results = []

    try:
        with open("sources.txt") as f:
            urls = [line.strip() for line in f if line.strip()]

        for url in urls:
            a = NewsArticle(url)
            try:
                a.download()
                a.parse()

                text = a.text or a.title
                sentiment_label, sentiment_score = get_sentiment(text)
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
                    "sentiment_score": round(sentiment_score, 3),
                })

            except Exception as e:
                results.append({
                    "title": f"FAILED to parse article. Error: {e}",
                    "source": urlparse(url).netloc,
                    "sentiment": "error",
                    "sentiment_score": 0,
                })

    except FileNotFoundError:
        results.append({
            "title": "Error: sources.txt not found.",
            "source": "N/A",
            "sentiment": "error",
            "sentiment_score": 0,
        })

    return results

# FLASK ROUTES Web Endpoints
# Homemenu route
@app.route("/")
def index():
    # Run batch analysis and render the results page
    analysis_results = run_sentiment_analysis()
    return render_template("results.html",
                           analysis_results=analysis_results,
                           has_new=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    # analyse submitted URL and preloaded articles
    url = request.form.get("url")

    if not url:
        return redirect(url_for("index"))

    try:
        new_result = analyze_single_url(url)
        batch_results = run_sentiment_analysis()
        analysis_results = [new_result] + batch_results

    except Exception as e:
        analysis_results = [{
            "title": f"FAILED to parse article. Error: {e}",
            "source": "N/A",
            "sentiment": "error",
            "sentiment_score": 0,
        }]

    return render_template("results.html",
                           analysis_results=analysis_results,
                           has_new=True)


@app.route("/history")
def history():
    results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).all()
    return render_template("history.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)

