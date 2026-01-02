"""
Topic model evaluation utilities.

Includes:
- Topic coherence
- Topic diversity
"""

from typing import List
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


# ---------- Coherence ----------
def compute_coherence(topics: List[List[str]], texts: List[List[str]]):
    """
    Compute topic coherence score (c_v).

    Args:
        topics: list of topics (each topic is list of words)
        texts: tokenized documents

    Returns:
        Coherence score
    """
    dictionary = Dictionary(texts)

    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    )

    return coherence_model.get_coherence()


# ---------- Topic diversity ----------
def compute_topic_diversity(topics: List[List[str]], top_k: int = 10):
    """
    Compute topic diversity score.

    Args:
        topics: list of topics
        top_k: number of words per topic to consider

    Returns:
        Topic diversity score
    """
    unique_words = set()
    total_words = 0

    for topic in topics:
        words = topic[:top_k]
        unique_words.update(words)
        total_words += len(words)

    return len(unique_words) / total_words
