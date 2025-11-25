
from flask import Flask, render_template
from newspaper import Article
from textblob import TextBlob
from urllib.parse import urlparse

# Initialize the Flask application
app = Flask(__name__)

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def run_sentiment_analysis():
    results = []
    try:
        with open("sources.txt") as f:
            urls = [line.strip() for line in f if line.strip()]

        for url in urls:
            a = Article(url)
            try:
                a.download()
                a.parse()
                text = a.text or a.title
                sentiment = get_sentiment(text)
                
                # Store the results
                results.append({
                    'title': a.title, 
                    'source': urlparse(url).netloc, 
                    'sentiment': sentiment
                })
            except Exception as e:
                # Store failure message
                results.append({
                    'title': f"FAILED to parse article. Error: {e}", 
                    'source': urlparse(url).netloc, 
                    'sentiment': 'error'
                })
    except FileNotFoundError:
        # Handle case where sources.txt is missing
        results.append({'title': "Error: sources.txt not found.", 'source': 'N/A', 'sentiment': 'error'})

    return results

@app.route('/')
def index():
   
    analysis_results = run_sentiment_analysis()
    return render_template('results.html', analysis_results=analysis_results)

if __name__ == '__main__':
    # Run the Flask web server
    app.run(debug=True)