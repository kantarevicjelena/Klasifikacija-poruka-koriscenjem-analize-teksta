# Klasifikacija poruka korišćenjem analize teksta

Ovaj projekat predstavlja implementaciju sistema za automatsku klasifikaciju e-mail poruka korišćenjem tehnika mašinskog učenja i obrade prirodnog jezika (NLP). Sistem analizira sadržaj mejlova i svrstava ih u odgovarajuće kategorije, uz mogućnost određivanja prioriteta poruke.

## Opis Projekta

Cilj projekta je razvoj inteligentnog sistema koji automatski klasifikuje e-mail poruke u sledeće kategorije:
Lično
Poslovno
Promocije
Obaveštenja
Spam

### Pored same klasifikacije, sistem može da proceni način obrade poruke:
Pročitati odmah
Pročitati kasnije
Arhivirati

### Projekat koristi kombinaciju:
NLP tehnika
TF-IDF reprezentacije teksta
Ručno definisanih karakteristika
Više modela mašinskog učenja

### Korišćene Tehnologije

Python
Scikit-learn
XGBoost
NLTK
Stanza
Pandas
NumPy

### Korišćeni fajlovi

extract.py - Skripta za ekstrakciju e-mail poruka iz .mbox fajla i konverziju u CSV format.
preprocess.py - Modul za predobradu teksta koji obuhvata čišćenje teksta, uklanjanje linkova, uklanjanje stop reči, detekciju jezika, lematizaciju i stemovanje. Podržani jezici: srpski, engleski
features.py - Implementacija pipeline-a za ekstrakciju karakteristika. Obuhvata TF-IDF nad rečima i karakterima, signalne karakteristike, numeričke karakteristike, obradu domena pošiljaoca.
models.py - Sadrži implementaciju i kreiranje klasifikacionih modela: Logistic Regression, Random Forest, XGBoost, SVM, MLPClassifier
train.py - Skripta za treniranje modela, validaciju, evaluaciju performansi, prikaz konfuzionih matrica.
predict.py - Omogućava testiranje modela preko terminala unosom novog e-maila.
Projekat VI.docx - Projektna dokumentacija sa detaljnom analizom pripreme podataka, arhitekture sistema, evaluacije modela, analiza grešaka, mogućnosti unapređenja.
