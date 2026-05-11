import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from scipy.stats import loguniform, randint

from preprocessing import preprocess
from features import build_pipeline
from models import build_classifier
from utils import (
    plot_confusion_matrix,
    export_top_senders_per_class,   # analitički izvoz
    export_top_terms_per_class,     # analitički izvoz
)

# =========================== Helpers ===========================

def _add_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje kolone:
      - 'from'   : kanonizovana iz 'from' ili 'sender' (ako ne postoji, prazno)
      - 'domain' : domen iz adrese po šablonu '@domen'
    """
    df = df.copy()

    if "from" in df.columns:
        s = df["from"].astype(str).fillna("")
    elif "sender" in df.columns:
        s = df["sender"].astype(str).fillna("")
    else:
        s = pd.Series([""] * len(df), index=df.index)

    dom = s.str.extract(r"@([^>\s]+)")[0].fillna("unknown")
    df["from"] = s
    df["domain"] = dom
    return df


def _param_distributions_for(model_name: str):
    """Raspodele za RandomizedSearchCV (brže, stohastički)."""
    if model_name == "logreg":
        return {"clf__C": loguniform(1e-1, 10)}
    if model_name == "svm":  # LinearSVC
        return {"clf__C": loguniform(1e-1, 10)}
    if model_name == "rf":
        return {
            "clf__n_estimators": randint(50, 200),
            "clf__max_depth": randint(3, 15),
        }
    if model_name == "mlp":
        return {
            "clf__alpha": loguniform(1e-4, 1e-2),
            "clf__hidden_layer_sizes": [(64,), (128,)],
            "clf__learning_rate_init": loguniform(1e-3, 1e-2),
            "clf__batch_size": [16, 32],
        }
    if model_name == "xgb":
        return {
            "clf__n_estimators": randint(50, 150),
            "clf__max_depth": randint(3, 6),
            "clf__learning_rate": loguniform(5e-2, 2e-1),
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    return None


def _param_grid_for(model_name: str):
    """Mali deterministički grid za GridSearchCV."""
    if model_name == "logreg":
        return {"clf__C": [0.1, 1.0, 10.0]}
    if model_name == "svm":  # LinearSVC
        return {"clf__C": [0.1, 1.0, 10.0]}
    if model_name == "rf":
        return {
            "clf__n_estimators": [150, 250, 350],
            "clf__max_depth": [10, 20, None],
        }
    if model_name == "mlp":
        return {
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__hidden_layer_sizes": [(128,), (256,), (256, 128)],
            "clf__learning_rate_init": [1e-4, 5e-4, 1e-3],
            "clf__batch_size": [32, 64],
        }
    if model_name == "xgb":
        return {
            "clf__n_estimators": [150, 250],
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    return None


def _maybe_wrap_with_search(pipeline, model_name: str, search_mode: str):
    """
    Vrati estimator u zavisnosti od search_mode: 'none' | 'random' | 'grid'.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if search_mode == "random":
        dists = _param_distributions_for(model_name)
        if not dists:
            return pipeline
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=dists,
            n_iter=5,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

    if search_mode == "grid":
        grid = _param_grid_for(model_name)
        if not grid:
            return pipeline
        return GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

    # search_mode == "none"
    return pipeline

# ======================= Glavna funkcija ========================

def train_many(
    data_path,
    model_names=None,
    search_mode="none",      # 'none' | 'random' | 'grid'
    out_dir="OUT",
    no_domain=False,         # ablacija: bez 'domain' feature-a
    no_split=False,          # treniraj na svemu (evaluacija na trainu)
):
    """
    Treniraj više modela u nizu, evaluiraj i sačuvaj:
      - OUT/model_{ime}.pkl                 : uvežban pipeline
      - OUT/cm_{ime}.png                    : matrica konfuzije (test ili train)
      - OUT/model_{ime}_labels.json (XGB)   : mapiranje indeks -> naziv klase
      - OUT/top_senders.csv i OUT/top_terms.csv : analitički izvoz (ako uspe)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Učitaj i pripremi podatke
    df = pd.read_csv(data_path)
    df = _add_domain(df)

    # Ako je greškom header u redu (label == "label") – ukloni
    if "label" in df.columns:
        bad_rows = df["label"] == "label"
        if bad_rows.any():
            df = df.loc[~bad_rows].copy()

    # Tekst (subject+body -> preprocess)
    df["text"] = (df.get("subject", "").fillna("") + " " +
                  df.get("body", "").fillna("")).apply(preprocess)

    # Analitički izvoz (ako postoje labele)
    try:
        if "label" in df.columns:
            export_top_senders_per_class(df[["label", "domain"]], out_csv=f"{out_dir}/top_senders.csv")
            export_top_terms_per_class(df["text"], df["label"], out_csv=f"{out_dir}/top_terms.csv")
            print(" Top pošiljaoci po klasi →", f"{out_dir}/top_senders.csv")
            print("Top termini po klasi →", f"{out_dir}/top_terms.csv")
    except Exception:
        print("[UPOZORENJE] Analitički izvoz nije uspeo (top_senders/top_terms).")

    # Ulazi za modele
    X_full = df[["text"]] if no_domain else df[["text", "domain"]]
    y = df["label"] if "label" in df.columns else None

    # Split
    if no_split or y is None:
        X_train_full, X_test_full = X_full, None
        y_train, y_test = y, None
        print("--no-split aktivan ili nema 'label' kolone: treniram na SVIM podacima (evaluacija na train setu).")
    else:
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_full, y, test_size=0.15, stratify=y, random_state=42
        )

    # Podrazumevani skup modela 
    if not model_names:
        model_names = ["logreg", "rf", "svm", "mlp", "xgb"]

    results = []

    # Petlja kroz tražene modele
    for name in model_names:
        print("\n" + "=" * 70)
        print(f"TRAINING: {name.upper()}")

        try:
            clf = build_classifier(name)
            pipeline = build_pipeline(clf, include_domain=not no_domain)

            # Umotaj pretragom (none/random/grid)
            pipeline = _maybe_wrap_with_search(pipeline, name, search_mode)

            # XGBoost: LabelEncoder + snimanje klasa
            if name == "xgb":
                if y_train is None:
                    raise ValueError("Za XGB je potreban skup sa 'label' kolonom (bar za treniranje).")

                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                pipeline.fit(X_train_full, y_train_enc)

                # evaluacija (test ako postoji, inače train)
                X_eval = X_test_full if X_test_full is not None else X_train_full
                if X_test_full is not None:
                    y_eval_enc = le.transform(y_test)
                else:
                    y_eval_enc = y_train_enc

                y_pred_enc = pipeline.predict(X_eval)
                y_pred = le.inverse_transform(y_pred_enc)
                y_eval = le.inverse_transform(y_eval_enc)

                rep = classification_report(y_eval, y_pred)
                print(rep)

                cm_png = f"{out_dir}/cm_{name}.png"
                plot_confusion_matrix(
                    y_eval, y_pred,
                    out_path=cm_png,
                    title=f"Confusion Matrix – {name}"
                )
                print(f"CM snimljen: {cm_png}")

                model_path = f"{out_dir}/model_{name}.pkl"
                joblib.dump(pipeline, model_path)
                print(f"Model snimljen: {model_path}")

                labels_path = f"{out_dir}/model_{name}_labels.json"
                with open(labels_path, "w", encoding="utf-8") as f:
                    json.dump(list(le.classes_), f, ensure_ascii=False)
                print(f"Sačuvane klase za XGB: {labels_path}")

                results.append((name, rep))
                continue

            # Ostali modeli
            if y_train is None:
                # treniraj na svemu i evaluiraj na svemu (optimistično)
                pipeline.fit(X_train_full, y)  # y je ceo niz labela
                y_pred = pipeline.predict(X_train_full)
                rep = classification_report(y, y_pred)
                print(rep)

                cm_png = f"{out_dir}/cm_{name}.png"
                plot_confusion_matrix(
                    y, y_pred,
                    out_path=cm_png,
                    title=f"Confusion Matrix – {name} (train)"
                )
                print(f"CM snimljen: {cm_png}")

                model_path = f"{out_dir}/model_{name}.pkl"
                joblib.dump(pipeline, model_path)
                print(f"Model snimljen: {model_path}")

                results.append((name, rep))

            else:
                pipeline.fit(X_train_full, y_train)
                X_eval = X_test_full if X_test_full is not None else X_train_full
                y_eval = y_test if X_test_full is not None else y_train

                y_pred = pipeline.predict(X_eval)
                rep = classification_report(y_eval, y_pred)
                print(rep)

                suffix = "test" if X_test_full is not None else "train"
                cm_png = f"{out_dir}/cm_{name}.png"
                plot_confusion_matrix(
                    y_eval, y_pred,
                    out_path=cm_png,
                    title=f"Confusion Matrix – {name} ({suffix})"
                )
                print(f"CM snimljen: {cm_png}")

                model_path = f"{out_dir}/model_{name}.pkl"
                joblib.dump(pipeline, model_path)
                print(f"Model snimljen: {model_path}")

                results.append((name, rep))

        except Exception as e:
            print(f"[GRESKA] {name}: {e}")
            continue

    # Sažetak
    print("\n" + "=" * 70)
    print("SAŽETAK MODELA:")
    for name, rep in results:
        print("-" * 40)
        print(name.upper())
        print(rep)
