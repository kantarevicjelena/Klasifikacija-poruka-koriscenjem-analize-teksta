from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# Optional: XGBoost (ostaje kao i do sada, ako je instaliran)
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # xgboost nije obavezan
    XGBClassifier = None  # type: ignore


def build_classifier(name: str = "logreg"):
    name = (name or "").lower()

    if name == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=10000,
            tol=1e-6,
            class_weight="balanced",
            random_state=42,
            C=1.0
        )
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200)
    elif name == "svm":
        return LinearSVC(C=1.0, class_weight="balanced")
    elif name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost nije instaliran. Pokreni: pip install xgboost")
        return XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

    # --- Novi: MLP (sklearn) ---
    elif name == "mlp":
        # Razumna podrazumevana konfiguracija za tekstualne TF-IDF featur-e
        return MLPClassifier(hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            max_iter=300,          # ↑ sa 200
            early_stopping=False,  # ostaje isključeno (rešava prethodni bug)
            n_iter_no_change=10,
            alpha=1e-4,            # blaga L2
            learning_rate_init=0.001,
            random_state=42
        )

    else:
        raise ValueError(f"Nepoznat klasifikator: {name}")