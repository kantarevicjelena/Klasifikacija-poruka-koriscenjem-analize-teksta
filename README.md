# Klasifikacija poruka korišćenjem analize teksta

Ovaj projekat predstavlja implementaciju sistema za automatsku klasifikaciju e-mail poruka korišćenjem tehnika mašinskog učenja i obrade prirodnog jezika (NLP). Sistem analizira sadržaj mejlova i svrstava ih u odgovarajuće kategorije, uz mogućnost određivanja prioriteta poruke.

## Opis Projekta

Cilj projekta je razvoj inteligentnog sistema koji automatski klasifikuje e-mail poruke u sledeće kategorije:
- Lično
- Poslovno
- Promocije
- Obaveštenja
- Spam

### Pored same klasifikacije, sistem može da proceni način obrade poruke:
- Pročitati odmah
- Pročitati kasnije
- Arhivirati

### Projekat koristi kombinaciju:
- NLP tehnika
- TF-IDF reprezentacije teksta
- Ručno definisanih karakteristika
- Više modela mašinskog učenja

### Korišćene Tehnologije

1. Python
2. Scikit-learn
3. XGBoost
4. NLTK
5. Stanza
6. Pandas
7. NumPy

### Korišćeni fajlovi

1. extract.py - Skripta za ekstrakciju e-mail poruka iz .mbox fajla i konverziju u CSV format.
2. preprocess.py - Modul za predobradu teksta koji obuhvata čišćenje teksta, uklanjanje linkova, uklanjanje stop reči, detekciju jezika, lematizaciju i stemovanje. Podržani jezici: srpski, engleski
3. features.py - Implementacija pipeline-a za ekstrakciju karakteristika. Obuhvata TF-IDF nad rečima i karakterima, signalne karakteristike, numeričke karakteristike, obradu domena pošiljaoca.
4. models.py - Sadrži implementaciju i kreiranje klasifikacionih modela: Logistic Regression, Random Forest, XGBoost, SVM, MLPClassifier
5. train.py - Skripta za treniranje modela, validaciju, evaluaciju performansi, prikaz konfuzionih matrica.
6. predict.py - Omogućava testiranje modela preko terminala unosom novog e-maila.
7. Projekat VI.docx - Projektna dokumentacija sa detaljnom analizom pripreme podataka, arhitekture sistema, evaluacije modela, analiza grešaka, mogućnosti unapređenja.
