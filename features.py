import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from joblib import Memory
from sklearn.neural_network import MLPClassifier

# Cache za ubrzanje tokom GridSearch/CV
CACHE_DIR = "cache"
memory = Memory(location=CACHE_DIR, verbose=0)

# --- Prošireni signali (ključne reči po klasama)
class SignalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        signals = []
        for text in X:
            signals.append([
                int(bool(re.search(r"unsubscribe|buy now|kliknite|registruj|spam|register|nagrada|price", text))),  # spam
                int(bool(re.search(r"discount|promo|akcija|popust|sale|kod|kupon|for you|za vas|bonus|join|poklon", text))),             # promocija
                int(bool(re.search(r"meeting|project|deadline|sastanak|kolege|zaposleni|izveštaj|plan|postovani|termin", text))),  # poslovno
                int(bool(re.search(r"family|birthday|weekend|love|zdravo|cao|hej|dragi|porodica|vikend|kafa|veceras|javi|rodjendan|zurka", text))),   # lično
                int(bool(re.search(r"update|notice|announcement|policy|obavestenje|izmena|pravila", text)))  # obaveštenje
            ])
        return np.array(signals)

# --- Numeričke karakteristike
class NumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        numeric = []
        for text in X:
            numeric.append([
                len(text),
                sum(1 for c in text if c.isupper()) / max(1, len(text)),
                text.count("http"),
            ])
        return np.array(numeric)

# --- TF-IDF za reči i karaktere
def build_vectorizer():
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=500, min_df=2, max_df=0.7, strip_accents="unicode"
    )
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 4), max_features=800, min_df=2, max_df=0.7
    )
    return FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec),
    ])

# --- Glavni pipeline sa opcionalnim 'domain' i skaliranjem za MLP
def build_pipeline(clf, include_domain: bool = True, svd_components: int = 100):
    text_union = FeatureUnion([
        ("tfidf", build_vectorizer()),
        ("signals", SignalFeatures()),
        ("numeric", NumericFeatures()),
    ])

    transformers = [
        ("text", text_union, "text"),
    ]

    if include_domain:
        transformers.append(
            ("domain", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["domain"])
        )

    features = ColumnTransformer(transformers)

    steps = [
        ("features", features),
    ]

    if svd_components and svd_components > 0:
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=42)))

    if isinstance(clf, MLPClassifier):
        steps.append(("scale", StandardScaler(with_mean=False)))

    steps.append(("clf", clf))
    pipeline = Pipeline(steps, memory=memory)
    return  pipeline