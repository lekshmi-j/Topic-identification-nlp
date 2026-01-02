"""
Topic modeling utilities.

Implements:
- LDA (Latent Dirichlet Allocation)
- NMF (Non-negative Matrix Factorization)
"""

from typing import List
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation, NMF


# ---------- LDA ----------
def train_lda(X_bow, num_topics: int, random_state: int = 42):
    """
    Train LDA model.

    Args:
        X_bow: Bag-of-Words document-term matrix
        num_topics: number of topics

    Returns:
        Trained LDA model
    """
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=10,
        learning_method="batch",
        random_state=random_state
    )
    lda.fit(X_bow)
    return lda


# ---------- NMF ----------
def train_nmf(X_tfidf, num_topics: int, random_state: int = 42):
    """
    Train NMF model.

    Args:
        X_tfidf: TF-IDF document-term matrix
        num_topics: number of topics

    Returns:
        Trained NMF model
    """
    nmf = NMF(
        n_components=num_topics,
        random_state=random_state
    )
    nmf.fit(X_tfidf)
    return nmf


# ---------- Topic extraction ----------
def extract_topics(model, feature_names: List[str], top_n: int = 10):
    """
    Extract top words per topic.

    Args:
        model: trained LDA or NMF model
        feature_names: vocabulary list
        top_n: number of top words per topic

    Returns:
        List of topics (list of words)
    """
    topics = []

    for topic in model.components_:
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-top_n - 1:-1]
        ]
        topics.append(top_words)

    return topics
