import os
import json
import re
import joblib
import pandas as pd
from preprocessing import preprocess
from utils import plot_confusion_matrix

# -------------------- MAPA AKCIJA --------------------
# Akcije na osnovu kategorije mejla
_ACTION_MAP = {
    "spam": "arhiviraj",
    "promocija": "procitaj kasnije",
    "obavestenje": "procitaj odmah",
    "poslovno": "prioritet - procitaj odmah",
    "licno": "procitaj kada stignes"
}


def _extract_domain(sender: str) -> str:
    sender = (sender or "").strip()
    m = re.search(r"@([^>\s]+)", sender)
    return m.group(1) if m else "unknown"


def _prepare_inputs_for_pipeline(pipeline, text: str, sender: str = ""):
    """Priprema input podatke za pipeline"""
    txt = preprocess(text or "")
    dom = _extract_domain(sender)
    return pd.DataFrame([{"text": txt, "domain": dom}])


def _maybe_inverse_labels(pred, labels_path: str):
    if not labels_path:
        return pred
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = list(json.load(f))
    import numpy as np
    pred = np.asarray(pred)
    return [classes[int(i)] for i in pred]


# -------------------- PREDICT ZA JEDAN TEKST --------------------
def predict_text(model_path: str, text: str, sender: str = "",
                 labels_path: str = None) -> str:
    """Predvidi kategoriju za jedan tekst"""
    # Input validation
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not text or not text.strip():
        return "nepoznato – procitaj odmah"

    pipeline = joblib.load(model_path)
    X = _prepare_inputs_for_pipeline(pipeline, text=text, sender=sender)

    # Predviđanje sa fallback logikom
    try:
        pred = pipeline.predict(X)
    except Exception:
        # Fallback - probaj samo sa text kolonom
        try:
            pred = pipeline.predict(X[["text"]])
        except Exception:
            pred = pipeline.predict(X["text"])

    pred = _maybe_inverse_labels(pred, labels_path)
    label = pred[0] if hasattr(pred, "_getitem_") else str(pred)

    # Jednostavan action mapping
    action = _ACTION_MAP.get(str(label).lower(), "procitaj odmah")
    return f"{label} – {action}"


# -------------------- PREDICT ZA CSV --------------------
def predict_csv(model_path, input_csv, output_csv,
                labels_path=None):
    """Predvidi kategorije za CSV file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    # Centralizovana domain extraction
    sender_col = "sender" if "sender" in df.columns else ("from" if "from" in df.columns else None)
    if sender_col is None:
        df["sender"] = ""
        sender_col = "sender"

    # Koristi centralizovanu funkciju za domain extraction
    df["domain"] = df[sender_col].apply(_extract_domain)

    # sklopi text i preprocess
    raw_text = (df.get("subject", "").fillna("") + " " + df.get("body", "").fillna("")).astype(str)
    df["text"] = raw_text.apply(preprocess)

    # Predviđanje sa graceful fallback
    try:
        y_pred_raw = model.predict(df[["text", "domain"]])
    except Exception:
        try:
            y_pred_raw = model.predict(df[["text"]])
        except Exception:
            y_pred_raw = model.predict(df["text"])

    # dekodiraj labele ako treba (XGB)
    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                classes = list(json.load(f))
            import numpy as np
            y_pred_raw = [classes[int(i)] for i in np.asarray(y_pred_raw)]
        except Exception:
            pass

    df["predicted_label"] = y_pred_raw

    # -------------------- DODAJ AKCIJE --------------------
    # Batch action mapping (za speed, bez confidence - to bi bilo skupo)
    df["action"] = [
        _ACTION_MAP.get(str(lbl).lower(), "procitaj odmah")
        for lbl in df["predicted_label"]
    ]

    # snimi rezultat
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predikcije sačuvane u {output_csv}")

    # Ako ima labelu → evaluacija
    if "label" in df.columns:
        from sklearn.metrics import classification_report
        rep = classification_report(df["label"], df["predicted_label"])
        print("\n===== CLASSIFICATION REPORT (na test CSV) =====")
        print(rep)

        cm_path = os.path.join(os.path.dirname(output_csv),
                               os.path.splitext(os.path.basename(output_csv))[0] + "_cm.png")
        plot_confusion_matrix(df["label"], df["predicted_label"], out_path=cm_path,
                              title="Confusion Matrix – Test CSV")
        print(f"🖼  Confusion matrix sačuvanu:{cm_path}")