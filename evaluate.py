"""
Quantitative Evaluation Script
Runs VADER, TextBlob, and Bias Analysis against a manually-labelled dataset
and reports Precision, Recall, F1-score per class and macro average.

Usage:
python evaluate.py

Output:
Per-engine classification report
Summary table suitable for thesis Chapter 5
"""

import csv
import os
from sklearn.metrics import classification_report, confusion_matrix

# import your own modules 
from ml_sentiment import get_vader_sentiment, get_textblob_sentiment
from bias_analysis import analyse_bias_language


# ──────────────────────────────────────────────────────────────────────
# Load the labelled dataset
# ──────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.csv")

def load_dataset(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Run evaluation
# ──────────────────────────────────────────────────────────────────────
def run_evaluation():
    dataset = load_dataset(DATASET_PATH)

    y_true_sentiment = []
    y_vader          = []
    y_textblob       = []

    y_true_bias      = []
    y_bias_pred      = []

    print(f"\nRunning evaluation on {len(dataset)} labelled samples...\n")

    for row in dataset:
        text     = row["text"]
        gt_sent  = row["ground_truth_sentiment"].strip().lower()
        gt_bias  = row["ground_truth_bias"].strip().lower()

        # Sentiment predictions
        vader_label,    _ = get_vader_sentiment(text)
        textblob_label, _ = get_textblob_sentiment(text)

        y_true_sentiment.append(gt_sent)
        y_vader.append(vader_label.lower())
        y_textblob.append(textblob_label.lower())

        # Bias prediction 
        bias_result = analyse_bias_language(text)
        y_true_bias.append(gt_bias)
        y_bias_pred.append(bias_result["bias_level"].lower())

    # Print results
    sentiment_labels = ["positive", "negative", "neutral"]
    bias_labels      = ["low", "moderate", "high"]

    print("=" * 60)
    print("VADER — Sentiment Classification")
    print("=" * 60)
    print(classification_report(
        y_true_sentiment, y_vader,
        labels=sentiment_labels,
        zero_division=0
    ))

    print("=" * 60)
    print("TextBlob — Sentiment Classification")
    print("=" * 60)
    print(classification_report(
        y_true_sentiment, y_textblob,
        labels=sentiment_labels,
        zero_division=0
    ))

    print("=" * 60)
    print("Bias Language Detector — Bias Level Classification")
    print("=" * 60)
    print(classification_report(
        y_true_bias, y_bias_pred,
        labels=bias_labels,
        zero_division=0
    ))

    # Confusion matrices 
    print("=" * 60)
    print("VADER Confusion Matrix  (rows=actual, cols=predicted)")
    print("Labels:", sentiment_labels)
    print(confusion_matrix(y_true_sentiment, y_vader, labels=sentiment_labels))

    print("\nTextBlob Confusion Matrix")
    print("Labels:", sentiment_labels)
    print(confusion_matrix(y_true_sentiment, y_textblob, labels=sentiment_labels))

    print("\nBias Level Confusion Matrix")
    print("Labels:", bias_labels)
    print(confusion_matrix(y_true_bias, y_bias_pred, labels=bias_labels))

    # Summary line
    from sklearn.metrics import f1_score, precision_score, recall_score

    def macro(y_true, y_pred, labels):
        p = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        r = recall_score   (y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f = f1_score       (y_true, y_pred, labels=labels, average="macro", zero_division=0)
        return round(p,3), round(r,3), round(f,3)

    vp, vr, vf = macro(y_true_sentiment, y_vader,    sentiment_labels)
    tp, tr, tf = macro(y_true_sentiment, y_textblob, sentiment_labels)
    bp, br, bf = macro(y_true_bias,      y_bias_pred, bias_labels)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE  (macro-averaged over all classes)")
    print("=" * 60)
    print(f"{'Engine':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    print(f"{'VADER (sentiment)':<30} {vp:>10.3f} {vr:>10.3f} {vf:>10.3f}")
    print(f"{'TextBlob (sentiment)':<30} {tp:>10.3f} {tr:>10.3f} {tf:>10.3f}")
    print(f"{'Bias Language Detector':<30} {bp:>10.3f} {br:>10.3f} {bf:>10.3f}")
    print("=" * 60)
    print("\nCopy the SUMMARY TABLE into your thesis Chapter 5 Evaluation section.")


if __name__ == "__main__":
    run_evaluation()
