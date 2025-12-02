from flask import Flask, render_template, request, redirect, url_for
from newspaper import Article as NewsArticle
from textblob import TextBlob
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


app = Flask(__name__)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


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


# Sentiment helper
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return label, polarity

def analyze_single_url(url):
    """Analyse one URL, save to DB, and return a result dict for the template."""
    a = NewsArticle(url)
    a.download()
    a.parse()

    text = a.text or a.title
    sentiment_label, sentiment_score = get_sentiment(text)
    source_domain = urlparse(url).netloc

    # DATABASE get or create Article row
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

    # DATABASE create AnalysisResult row 
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
        "sentiment": sentiment_label,
        "sentiment_score": round(sentiment_score, 3),
    }


# Main Analysis Logic

def run_sentiment_analysis():
    results = []

    try:
        # Read URLs from sources.txt
        with open("sources.txt") as f:
            urls = [line.strip() for line in f if line.strip()]

        for url in urls:
            # Use newspaper3k Article
            a = NewsArticle(url)
            try:
                a.download()
                a.parse()

                text = a.text or a.title
                sentiment_label, sentiment_score = get_sentiment(text)
                source_domain = urlparse(url).netloc

                # DATABASE get or create Article row
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

                # DATABASE create AnalysisResult row 
                analysis = AnalysisResult(
                    article_id=article_row.id,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                )
                db.session.add(analysis)
                db.session.commit()

                # Sends to template
                results.append({
                    "title": a.title,
                    "source": source_domain,
                    "sentiment": sentiment_label,
                    "sentiment_score": round(sentiment_score, 3),
                })

            except Exception as e:
                results.append({
                    "title": f"FAILED to parse article. Error: {e}",
                    "source": urlparse(url).netloc,
                    "sentiment": "error",
                })

    except FileNotFoundError:
        results.append({
            "title": "Error: sources.txt not found.",
            "source": "N/A",
            "sentiment": "error",
        })

    return results

# Routes
@app.route("/")
def index():
    analysis_results = run_sentiment_analysis()
    return render_template("results.html", analysis_results=analysis_results)

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url")

    if not url:
        # No URL provided, just go back to the main page
        return redirect(url_for("index"))

    try:
        # analyse the URL the user entered
        new_result = analyze_single_url(url)

        # also run the normal batch analysis from sources.txt
        batch_results = run_sentiment_analysis()

        # put the new one at the top
        analysis_results = [new_result] + batch_results
        
    except Exception as e:
        # If anything goes wrong, show a single error row
        analysis_results = [{
            "title": f"FAILED to parse article. Error: {e}",
            "source": "N/A",
            "sentiment": "error",
            "sentiment_score": 0,
        }]

    return render_template("results.html", analysis_results=analysis_results)


@app.route("/history")
def history():
    results = AnalysisResult.query.all()
    return render_template("history.html", results=results)

# Run Server
if __name__ == "__main__":
    app.run(debug=True)