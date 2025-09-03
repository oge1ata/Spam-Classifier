# Spam-Classifier
Spam SMS/Email Classifier (N-grams + ROC/AUC) By Ogechukwu Ata

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
