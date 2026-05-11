import re
from typing import List

# --- Osnovne stop-reči (SR + mali EN skup)
STOPWORDS_SR = {"i","ili","ali","da","je","sam","si","su","će","ce","bi","se","na","u","o","od","za","do","po","koji","što","sto"}
STOPWORDS_EN = {"the","a","an","and","or","but","to","of","in","on","for","with","is","are","was","were","be","been","am","by","at","as"}
STOPWORDS = STOPWORDS_SR | STOPWORDS_EN

# --- Jezička heuristika (brza, bez biblioteka)
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_HAS_SR_DIACRITICS = re.compile(r"[čćžšđČĆŽŠĐ]")

# --- langdetect fallback
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _langdetect_ok = True
except Exception:
    _langdetect_ok = False


def _detect_lang(s: str) -> str:
    s = s or ""
    # 1) Ako ima ćirilicu ili dijakritike → sigurno srpski
    if _CYRILLIC.search(s) or _HAS_SR_DIACRITICS.search(s):
        return "sr"

    # 2) Inače, heuristika preko stop-reči (ASCII varijante srpskog)
    tokens = re.findall(r"[a-zA-Z]+", s.lower())
    sr_ascii = {
        "da","je","sam","si","smo","ste","su","se","ti","mi","vi","oni","one","ono",
        "u","na","o","od","za","do","po","kod","pri","pre","posle","bez","sa",
        "i","ili","ali","nego","no",
        "koji","koja","koje","kojima","sto",
        "taj","ta","to","ovaj","ova","ovo","onaj","ona","ono",
        "ne","nisam","nisi","nije","nismo","niste","nisu"
    }
    sr_count = sum(t in (STOPWORDS_SR | sr_ascii) for t in tokens)
    en_count = sum(t in STOPWORDS_EN for t in tokens)

    if sr_count >= en_count and (sr_count + en_count) > 0:
        return "sr"

    # 3) Ako heuristika nije odlučna → koristi langdetect ako postoji
    if _langdetect_ok:
        try:
            lang = detect(s)
            if lang.startswith("sr"):
                return "sr"
            elif lang.startswith("en"):
                return "en"
            else:
                return "en"  # fallback
        except Exception:
            return "en"
    return "en"

# --- Pokušaj import-a lematizatora (graceful fallback)
_nltk_ok = False
_stanza_ok = False

# EN: NLTK WordNetLemmatizer (fallback: Porter)
try:
    import nltk  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    from nltk.stem import PorterStemmer  # type: ignore
    _wnl = WordNetLemmatizer()
    _porter = PorterStemmer()
    _nltk_ok = True
except Exception:
    _wnl = None
    _porter = None

# SR: Stanza lematizator (sr model mora biti preuzet: stanza.download('sr'))
try:
    import stanza  # type: ignore
    try:
        _stanza_sr = stanza.Pipeline("sr", processors="tokenize,pos,lemma", tokenize_pretokenized=False, use_gpu=False)
        _stanza_ok = True
    except Exception:
        _stanza_sr = None
        _stanza_ok = False
except Exception:
    _stanza_sr = None
    _stanza_ok = False


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"[^a-zA-ZčćžšđČĆŽŠĐ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lemmatize_en(tokens: List[str]) -> List[str]:
    if not _nltk_ok:
        return tokens
    out = []
    for t in tokens:
        try:
            w = _wnl.lemmatize(t, pos="v")
            w = _wnl.lemmatize(w, pos="n")
        except Exception:
            w = _porter.stem(t) if _porter else t
        out.append(w)
    return out


def _lemmatize_sr(text: str) -> List[str]:
    if not _stanza_ok or _stanza_sr is None:
        return text.split()
    try:
        doc = _stanza_sr(text)
        lemmas: List[str] = []
        for sent in doc.sentences:
            for w in sent.words:
                lemmas.append(w.lemma or w.text)
        return lemmas
    except Exception:
        return text.split()


def preprocess(text: str, use_lemmatization: bool = True) -> str:
    text = clean_text(text)
    lang = _detect_lang(text)

    if not use_lemmatization:
        tokens = [t for t in text.split() if t not in STOPWORDS]
        return " ".join(tokens)

    if lang == "sr":
        lemmas = _lemmatize_sr(text)
        tokens = [t for t in lemmas if t not in STOPWORDS]
        return " ".join(tokens)
    else:
        tokens = [t for t in text.split() if t not in STOPWORDS]
        tokens = _lemmatize_en(tokens)
        tokens = [t for t in tokens if t and t not in STOPWORDS]
        return " ".join(tokens)