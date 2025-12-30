"""
Text preprocessing utilities for topic identification.

This module provides a reusable preprocessing pipeline including:
- lowercasing
- tokenization
- stopword removal
- lemmatization
- bigram generation
- rare word filtering
- vocabulary size limiting
"""

import re
from collections import Counter
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Phrases
from gensim.models.phrases import Phraser


# ---------- NLTK setup ----------
def download_nltk_resources():
    """Download required NLTK resources (call once)."""
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# ---------- Basic preprocessing ----------
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def basic_preprocess(text: str) -> List[str]:
    """
    Apply basic preprocessing to a single document.

    Steps:
    - lowercase
    - remove non-alphabetic characters
    - tokenize
    - remove stopwords
    - lemmatize

    Returns:
        List of cleaned tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return tokens


# ---------- Bigram modeling ----------
def apply_bigrams(
    tokenized_docs: List[List[str]],
    min_count: int = 10,
    threshold: int = 15,
) -> List[List[str]]:
    """
    Learn and apply bigram phrases to tokenized documents.

    Args:
        tokenized_docs: list of token lists
        min_count: minimum frequency for bigram formation
        threshold: phrase detection threshold

    Returns:
        Tokenized documents with bigrams applied
    """
    phrases = Phrases(
        tokenized_docs,
        min_count=min_count,
        threshold=threshold,
    )
    phraser = Phraser(phrases)

    return [phraser[doc] for doc in tokenized_docs]


# ---------- Vocabulary filtering ----------
def filter_by_frequency(
    tokenized_docs: List[List[str]],
    min_freq: int = 20,
    max_vocab_size: int = 8000,
) -> List[List[str]]:
    """
    Remove rare words and limit vocabulary size.

    Args:
        tokenized_docs: list of token lists
        min_freq: minimum frequency required to keep a token
        max_vocab_size: maximum allowed vocabulary size

    Returns:
        Filtered tokenized documents
    """
    all_tokens = [t for doc in tokenized_docs for t in doc]
    freq = Counter(all_tokens)

    # keep tokens above min frequency
    filtered_docs = [
        [t for t in doc if freq[t] >= min_freq]
        for doc in tokenized_docs
    ]

    # limit vocabulary size
    most_common_tokens = {
        word for word, _ in freq.most_common(max_vocab_size)
    }

    filtered_docs = [
        [t for t in doc if t in most_common_tokens]
        for doc in filtered_docs
    ]

    return filtered_docs


# ---------- Full pipeline ----------
def preprocess_corpus(
    texts: List[str],
    min_bigram_count: int = 10,
    bigram_threshold: int = 15,
    min_token_freq: int = 20,
    max_vocab_size: int = 8000,
) -> List[List[str]]:
    """
    Full preprocessing pipeline for topic modeling.

    Args:
        texts: list of raw documents
        min_bigram_count: min frequency for bigram creation
        bigram_threshold: threshold for bigram detection
        min_token_freq: minimum token frequency
        max_vocab_size: maximum vocabulary size

    Returns:
        List of cleaned token lists
    """
    # basic cleaning
    tokenized_docs = [basic_preprocess(text) for text in texts]

    # bigram modeling
    tokenized_docs = apply_bigrams(
        tokenized_docs,
        min_count=min_bigram_count,
        threshold=bigram_threshold,
    )

    # frequency & vocabulary filtering
    tokenized_docs = filter_by_frequency(
        tokenized_docs,
        min_freq=min_token_freq,
        max_vocab_size=max_vocab_size,
    )

    return tokenized_docs
