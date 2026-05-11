import os
import argparse
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# -------------------- helperi --------------------

def _ensure_dirs(path: str):
    if path and len(path.strip()) > 0:
        os.makedirs(path, exist_ok=True)

def _canon_sender_col(df: pd.DataFrame) -> pd.DataFrame:
    """Obezbedi kolone 'sender' i 'sender_domain'."""
    df = df.copy()
    if "sender" in df.columns:
        s = df["sender"].astype(str)
    elif "from" in df.columns:
        s = df["from"].astype(str)
    else:
        s = pd.Series([""] * len(df), index=df.index)
    df["sender"] = s
    df["sender_domain"] = df["sender"].apply(
        lambda x: x.split("@")[1].lower() if "@" in str(x) else "unknown"
    )
    return df

def _drop_header_label_row(df: pd.DataFrame) -> pd.DataFrame:
    """Ako postoji red sa label == 'label' (greška u CSV-u) – ukloni ga."""
    if "label" in df.columns:
        bad = df["label"] == "label"
        if bad.any():
            df = df.loc[~bad].copy()
    return df

def _savefig(fig, out_dir: str, filename: str) -> str:
    _ensure_dirs(out_dir)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return path

# -------------------- 1) Učitavanje i osnovna analiza --------------------

def load_and_analyze_data(csv_path: str) -> pd.DataFrame:
    print("=== UČITAVANJE I ANALIZA PODATAKA ===")
    df = pd.read_csv(csv_path)
    df = _drop_header_label_row(df)
    df = _canon_sender_col(df)

    print(f"📊 Ukupan broj poruka: {len(df)}")
    print(f"📋 Kolone: {list(df.columns)}")

    if "label" in df.columns:
        class_dist = df["label"].value_counts()
        print("\n📈 Distribucija po klasama:")
        for label, count in class_dist.items():
            pct = (count / max(1, len(df))) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    else:
        print(" Nema kolone 'label' – deo vizualizacija će biti preskočen.")

    return df

# -------------------- 2) Vizualizacije (očišćeno) --------------------

def create_visualizations(df: pd.DataFrame, out_dir: str):
    print("\n=== KREIRANJE VIZUALIZACIJA ===")

    plt.style.use("default")
    sns.set_palette("husl")

    df = df.copy()
    df["subject"] = df.get("subject", "").fillna("")
    df["body"] = df.get("body", "").fillna("")
    df["message_length"] = df["subject"].str.len() + df["body"].str.len()
    df["word_count"] = df.apply(
        lambda r: len(str(r["subject"]).split()) + len(str(r["body"]).split()),
        axis=1,
    )
    df["exclamation_count"] = df["subject"].str.count("!") + df["body"].str.count("!")
    df["question_count"] = df["subject"].str.count(r"\?") + df["body"].str.count(r"\?")

    paths = {}

    # 1) Pie chart distribucija klasa (zadržavamo)
    if "label" in df.columns:
        class_counts = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            class_counts.values,
            labels=class_counts.index,
            autopct="%1.1f%%",
            colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"],
            startangle=90,
        )
        ax.set_title("Udeo klasa (pie)", fontweight="bold")
        paths["class_dist_pie"] = _savefig(fig, out_dir, "class_dist_pie.png")
        labels_order = list(class_counts.index)
    else:
        labels_order = None

    # 2) Top domeni pošiljalaca (zadržavamo)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_domains = df["sender_domain"].value_counts().head(10)
    ax.bar(
        range(len(top_domains)),
        top_domains.values,
        color=plt.cm.Set3(np.linspace(0, 1, len(top_domains))),
    )
    ax.set_xticks(range(len(top_domains)))
    ax.set_xticklabels(top_domains.index, rotation=35, ha="right")
    ax.set_title("Top domeni pošiljalaca", fontweight="bold")
    ax.set_ylabel("Broj poruka")
    paths["sender_domains"] = _savefig(fig, out_dir, "sender_domains.png")

    # 3) Statistike + prosečan broj reči po klasi (zadržavamo SAMO avg words)
    if labels_order:
        stats = (
            df.groupby("label")
            .agg(
                word_count_mean=("word_count", "mean"),
                word_count_std=("word_count", "std"),
                exclamation_mean=("exclamation_count", "mean"),
                question_mean=("question_count", "mean"),
            )
            .round(2)
        )
        print("\n Statistike po klasama:")
        print(stats)

        # Prosečan broj reči po klasi (zadržavamo)
        fig, ax = plt.subplots(figsize=(9, 6))
        means = stats["word_count_mean"].reindex(labels_order)
        bars = ax.bar(means.index, means.values, color="#96CEB4", alpha=0.9)
        ax.set_title("Prosečan broj reči po klasi", fontweight="bold")
        ax.set_ylabel("Broj reči")
        ax.tick_params(axis="x", rotation=35)
        for b, val in zip(bars, means.values):
            ax.text(
                b.get_x() + b.get_width() / 2.,
                b.get_height() + 0.8,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        paths["avg_words_by_label"] = _savefig(fig, out_dir, "avg_words_by_label.png")

    print("\n Sačuvani grafikoni:")
    for k, p in paths.items():
        print(f"  • {k}: {p}")

    return paths

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Vizuelizacije za projekat VI (očišćena verzija)")
    parser.add_argument("--data", default="dataset_balanced_saTest.csv",
                        help="Putanja do CSV fajla sa podacima")
    parser.add_argument("--out", default="OUT/figs",
                        help="Direktorijum gde se čuvaju PNG grafici")
    args = parser.parse_args()

    print("POKRETANJE VIZUELNE ANALIZE")
    print("=" * 60)

    df = load_and_analyze_data(args.data)
    create_visualizations(df, args.out)

    print("\nVIZUELIZACIJE GENERISANE. PNG fajlovi su u:", os.path.abspath(args.out))
    print("=" * 60)

if __name__ == "__main__":
    main()
