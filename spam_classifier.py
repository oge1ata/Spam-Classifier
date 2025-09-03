"""
Spam SMS/Email Classifier (N-grams + ROC/AUC) By Ogechukwu Ata
---------------------------------------------

What this script does:
1) Loads the SMS Spam dataset (UCI TSV or a CSV with columns: label, text).
2) Preprocesses text with N-grams (unigrams+bigrams or trigrams).
3) Trains and compares two models:
   - Logistic Regression with TF-IDF features
   - Multinomial Naive Bayes with Count features
4) Evaluates with ROC curves and AUC (plus accuracy, F1, confusion matrix).
5) Saves the best model to disk (joblib).

Run:
    python spam_classifier.py

Notes:
- Place UCI file at data/SMSSpamCollection (tab-separated).
- OR place a CSV at data/sms_spam.csv with headers: label,text
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from joblib import dump

DATA_DIR = Path("data")
UCI_PATH = DATA_DIR / "SMSSpamCollection"     # tab-separated: label \t text
CSV_PATH = DATA_DIR / "spam.csv"          # optional CSV fallback (label,text)

RANDOM_STATE = 42

def load_dataset():
    if UCI_PATH.exists():
        df = pd.read_csv(UCI_PATH, sep="\t", header=None, names=["label", "text"])
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
        # Normalizing the column names
        df = df.rename(columns={"v1": "label", "v2": "text"})
        df.columns = [c.strip().lower() for c in df.columns]
        if not {"label", "text"}.issubset(df.columns):
            raise ValueError("CSV must contain 'label' and 'text' columns")
    else:
        raise FileNotFoundError("Dataset not found")

    #keeping only two columns and dropping obvious noise
    df = df[["label", "text"]].dropna()

    #basic cleanup of labels
    df["label"] = df["label"].str.strip().str.lower()
    return df

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Build two pipelines and compare:
    1) TF-IDF (1–2 grams) + Logistic Regression
    2) Count (1–2 grams) + Multinomial Naive Bayes
    Evaluate with ROC/AUC primarily, also accuracy/F1, and plot ROC curves.
    """
    # Pipelines
    pipe_lr = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None, random_state=RANDOM_STATE)),
    ])
    # Explanation: pipe_lr is a sklearn Pipeline with two steps:
    #  - "tfidf": convert raw text -> TF-IDF numeric features using 1-grams and 2-grams
    #  - "clf": LogisticRegression classifier that consumes the TF-IDF features

    pipe_nb = Pipeline([
        ("count", CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    # Explanation: pipe_nb uses raw token counts (CountVectorizer) + MultinomialNB

    models = {
        "TFIDF+LogReg": pipe_lr,
        "Count+MultinomialNB": pipe_nb,
    }
    # Put the two pipelines in a dictionary so we can iterate over them

    results = []
    # We'll append performance summaries (dicts) for each model to this list

    plt.figure(figsize=(7, 6))
    # Prepare a matplotlib figure for ROC curves

    for name, model in models.items():
        model.fit(X_train, y_train)
        # Fit the pipeline: Vectorizer learns vocabulary then the classifier is trained

        # Probabilities for positive class (spam=1)
        y_proba = model.predict_proba(X_test)[:, 1]
        # predict_proba returns shape (n_samples, n_classes). [:,1] is the probability for class "1" (spam)

        y_pred = (y_proba >= 0.5).astype(int)
        # Convert probabilites to hard labels using threshold 0.5

        auc = roc_auc_score(y_test, y_proba)
        # AUC uses the continuous scores (y_proba), not the hard labels

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Standard classification metrics computed from the hard predictions

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        # roc_curve returns false-positive rates and true-positive rates for many thresholds

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        # Plot this model's ROC curve on the shared figure

        results.append({
            "model": name,
            "AUC": auc,
            "Accuracy": acc,
            "F1": f1,
        })
        # Store summary metrics for later comparison

        print(f"\n=== {name} ===")
        print(f"AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
        print("Confusion matrix [[TN FP][FN TP]]:")
        print(confusion_matrix(y_test, y_pred))
        # Print readable metrics and confusion matrix for inspection

    # After the loop, finalize the ROC plot
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    # Diagonal baseline = random classifier
    plt.title("ROC Curves (Spam vs Ham)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    # Display the ROC figure

    # Pick best by AUC
    best = max(results, key=lambda r: r["AUC"])
    # results is a list of dicts; choose the dict with highest 'AUC'

    best_model = models[best["model"]]
    # Map the best name back to the pipeline object (this pipeline was already fit above)
    return best_model, results


def main():
    df = load_dataset()
    print("Sample:")
    print(df.head())

    # Map labels to binary: ham=0, spam=1
    lb = LabelBinarizer(pos_label=1, neg_label=0)
    y = lb.fit_transform(df["label"]).ravel()  # ham->0, spam->1
    X = df["text"].astype(str)

    # Train/test split (stratified to preserve spam/ham ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    best_model, results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nModel comparison (by AUC, Accuracy, F1):")
    print(pd.DataFrame(results).sort_values("AUC", ascending=False).reset_index(drop=True))

    # Refit best on full data (optional, for deployment)
    best_model.fit(X, y)
    Path("models").mkdir(exist_ok=True)
    model_path = Path("models") / "spam_best_model.joblib"
    dump(best_model, model_path)
    print(f"\nSaved best model to: {model_path.resolve()}")

if __name__ == "__main__":
    main()


"""
Plain-English walkthrough of what happens (stepwise)

Create two pipelines:

pipe_lr: TfidfVectorizer → LogisticRegression.

TF-IDF transforms each message into a numeric vector that downweights common words and upweights informative words/phrases.

pipe_nb: CountVectorizer → MultinomialNB.

CountVectorizer produces raw counts of tokens/ngrams. MultinomialNB expects counts and models word frequencies directly.

Put pipelines in a dictionary so you can loop through both with the same code, keeping output consistent.

Prepare the ROC plot with plt.figure(...) — both ROC curves will be drawn on the same axes.

For each model:

model.fit(X_train, y_train):

The pipeline first fits the vectorizer (learns vocabulary and IDF if TF-IDF), then fits the classifier on the transformed training data.

model.predict_proba(X_test)[:, 1]:

Get probability that each test sample is class 1 (spam). Using probabilities lets you compute ROC/AUC which measures ranking quality across thresholds.

y_pred = (y_proba >= 0.5).astype(int):

Convert probabilities to binary predictions at threshold 0.5 so we can compute accuracy, F1, and the confusion matrix.

Compute metrics:

roc_auc_score(y_test, y_proba) — AUC over full range of thresholds (0..1).

accuracy_score and f1_score — depend on chosen threshold (0.5 here).

roc_curve returns arrays (fpr, tpr, thresholds) used to draw the ROC curve.

Append a summary dict to results for later comparison and print a human-readable report.

After loop:

Add diagonal baseline to ROC plot (random guess).

Add labels and show the figure.

Pick the best model by highest AUC:

best = max(results, key=lambda r: r["AUC"]) finds the summary dict with the best AUC.

best_model = models[best["model"]] retrieves the corresponding pipeline object (which, remember, was fitted earlier in the loop).

Return the fitted best pipeline and the results list.

One practical note: the returned best_model is the pipeline as trained on X_train (i.e., trained only on the training fold). If you want a model trained on the entire dataset for production, call best_model.fit(X, y) on your full data (that’s what your script does after).

Important details & gotchas

predict_proba(... )[:, 1] assumes the second column corresponds to the positive class (label 1). That’s true here because labels were encoded as 0 (ham) and 1 (spam). If your label encoding is different, check the classes_ attribute of the classifier.

min_df=2 in the vectorizers removes terms that appear in fewer than 2 documents — this reduces noise and shrink vocabulary.

stop_words="english" removes common English words; sometimes useful, sometimes you’ll want to keep them if they carry meaning for your task.

max_iter=2000 for LogisticRegression increases iterations to ensure convergence for high-dimensional sparse TF-IDF inputs.

The pipeline object stored in models was mutated (fitted) inside the loop — retrieving it later returns the fitted instance, not a fresh untrained copy.
"""
