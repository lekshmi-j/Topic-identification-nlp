"""
Supervised text classification utilities.

Implements:
- TF-IDF vectorization
- Logistic Regression
- Naive Bayes classifiers
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def build_tfidf(
    max_df=0.5,
    min_df=10,
    ngram_range=(1, 2)
):
    """
    Create a TF-IDF vectorizer.
    """
    return TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range
    )


def train_logistic_regression(X, y):
    """
    Train Logistic Regression classifier.
    """
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def train_naive_bayes(X, y):
    """
    Train Multinomial Naive Bayes classifier.
    """
    model = MultinomialNB()
    model.fit(X, y)
    return model
