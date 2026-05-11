# main.py
import argparse
from train import train_many
from predict import predict_text, predict_csv

def main():
    parser = argparse.ArgumentParser(description="Projekat VI – CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract-mbox (leni import jer nije svima potrebno)
    p_ext = sub.add_parser("extract-mbox", help="Izvuci poruke iz .mbox u CSV")
    p_ext.add_argument("--mbox", required=True)
    p_ext.add_argument("--csv", required=True)

    # train-all
    p_train_all = sub.add_parser("train-all", help="Treniraj više modela i prikaži izveštaj za svaki")
    p_train_all.add_argument("--data", required=True, help="CSV sa kolonama subject, body, (sender|from), label")
    p_train_all.add_argument(
        "--models",
        default="logreg,rf,svm,mlp,xgb",
        help="Lista modela zarezom: logreg,rf,svm,mlp,xgb",
    )
    p_train_all.add_argument(
        "--search",
        choices=["none", "random", "grid"],
        default="none",
        help="Hiperparametarska pretraga: none | random | grid",
    )
    p_train_all.add_argument("--out", default="OUT", help="Direktorijum za modele i vizualizacije")
    p_train_all.add_argument("--no-domain", action="store_true", help="Ablacija: ne koristi domen pošiljaoca")
    p_train_all.add_argument("--no-split", action="store_true", help="Treniraj na svim podacima (evaluacija na train)")

    # predict (jedan tekst)
    p_pred = sub.add_parser("predict", help="Predikcija jednog teksta")
    p_pred.add_argument("--model", required=True, help="Putanja do .pkl pipeline modela")
    p_pred.add_argument("--text", required=False, help="Ceо tekst poruke (ako ne koristiš sender/subject/body)")
    p_pred.add_argument("--sender", required=False, default="", help="Npr. 'Ime <email@domen.com>'")
    p_pred.add_argument("--subject", required=False, default="")
    p_pred.add_argument("--body", required=False, default="")
    p_pred.add_argument("--labels", required=False, help="JSON sa klasama (za XGB i sl.)")

    # predict-csv (batch)
    p_predcsv = sub.add_parser("predict-csv", help="Predikcija nad CSV fajlom")
    p_predcsv.add_argument("--model", required=True)
    p_predcsv.add_argument("--input", required=True)
    p_predcsv.add_argument("--output", required=True)
    p_predcsv.add_argument("--labels", required=False, help="JSON sa klasama (za XGB i sl.)")

    args = parser.parse_args()

    if args.cmd == "extract-mbox":
        import extract as ex
        if hasattr(ex, "extract_mbox_to_csv"):
            ex.extract_mbox_to_csv(args.mbox, args.csv)
        elif hasattr(ex, "extract_to_csv"):
            ex.extract_to_csv(args.mbox, args.csv)
        else:
            raise ImportError("U 'extract.py' nema funkcija 'extract_mbox_to_csv' ili 'extract_to_csv'.")

    elif args.cmd == "train-all":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        train_many(
            data_path=args.data,
            model_names=models,
            search_mode=args.search,   # none | random | grid
            out_dir=args.out,
            no_domain=args.no_domain,
            no_split=args.no_split,
        )

    elif args.cmd == "predict":
        # Ako je prosleđen ceo tekst, koristi to; inače sklopi iz subject + body
        text = args.text
        if not text:
            text = f"{args.subject or ''} {args.body or ''}".strip()
        print(predict_text(
            model_path=args.model,
            text=text,
            sender=args.sender or "",
            labels_path=args.labels
        ))

    elif args.cmd == "predict-csv":
        predict_csv(
            model_path=args.model,
            input_csv=args.input,
            output_csv=args.output,
            labels_path=args.labels
        )


if __name__ == "__main__":
    main()
