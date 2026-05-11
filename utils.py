import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2


def plot_confusion_matrix(y_true, y_pred, labels=None,
                          out_path="OUT/confusion_matrix.png",
                          title="Confusion Matrix"):
    # Ako labels nije prosleđen, izračunaj ga iz podataka (unija stvarnih i predikovanih klasa)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Napravi OUT folder ako ne postoji
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.title(title)
    plt.xlabel("Predikcija")
    plt.ylabel("Tačno")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# (D) Top pošiljaoci po klasi – CSV
def export_top_senders_per_class(df_label_domain: pd.DataFrame,
                                 out_csv: str, top_n: int = 10):
    """Očekuje kolone: label, domain."""
    if not {"label", "domain"}.issubset(df_label_domain.columns):
        raise ValueError("Potrebne kolone: 'label' i 'domain'.")

    rows = []
    for lbl, sub in df_label_domain.groupby("label"):
        top = sub["domain"].value_counts().head(top_n).reset_index()
        top.columns = ["domain", "count"]
        top.insert(0, "label", lbl)
        rows.append(top)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
    print(f" Top pošiljaoci po klasi → {out_csv}")


# (D) Top termini po klasi – CSV (chi2 nad zasebnim TF-IDF-om bez SVD)
def export_top_terms_per_class(texts: pd.Series, labels: pd.Series,
                               out_csv: str, top_n: int = 20):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode")
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()

    rows = []
    for lbl in sorted(pd.unique(labels)):
        y_bin = (labels == lbl).astype(int)
        scores, _ = chi2(X, y_bin)
        top_idx = scores.argsort()[::-1][:top_n]
        for i in top_idx:
            rows.append({"label": lbl, "term": terms[i], "chi2": float(scores[i])})

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Top termini po klasi → {out_csv}")
