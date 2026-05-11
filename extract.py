import mailbox
import pandas as pd
from email.utils import parsedate_to_datetime

def extract_mbox_to_csv(mbox_file, csv_file):
    mbox = mailbox.mbox(mbox_file)
    rows = []
    for msg in mbox:
        subject = (msg.get("subject") or "")
        sender  = (msg.get("from") or "")
        date_raw = msg.get("date")
        try:
            date = parsedate_to_datetime(date_raw).isoformat() if date_raw else ""
        except Exception:
            date = date_raw or ""

        # Telo: robustno izvlačenje (multipart ili ne)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                    except Exception:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                body = msg.get_payload() or ""

        # label ručno naknadno
        rows.append([sender, subject, body, date, "unknown"])

    df = pd.DataFrame(rows, columns=["sender", "subject", "body", "date", "label"])
    df.to_csv(csv_file, index=False)
    print(f"Ekstrahovano {len(df)} mejlova u {csv_file}")
