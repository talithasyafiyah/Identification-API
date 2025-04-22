import json
import re
from html import unescape
import numpy as np
from gensim.models import FastText
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load slang dictionary
with open("final_slang.txt", "r", encoding="utf-8") as f:
    SLANG_DICT = json.load(f)

# Stopwords
factory = StopWordRemoverFactory()
STOPWORDS = set(factory.get_stop_words()) | {
    "huhu", "wkwk", "dong", "akwkakakak", "hehe", "akxkakskak", "sikxskxk",
    "wkwkwkwk", "wkwkwk", "wkakka", "wkkw", "wkwkw", "wkwkwkwk", "wkwkwkwkwk",
    "hehehe", "heheh", "dongg", "wkwwkwk", "wkwkwkkw", "wkwkwkw", "awokwkwkwkw",
    "aowkwoowkwwkwkkw", "wkwkwwk", "wkwkkww", "wkwkk", "wkwkwkk", "hehehaha",
    "ehehehe", "mwehehe", "hihi", "wkwkak", "wkwkwkkwk", "wkwkwkwkwkwkwk",
    "kwkwkw", "wkwkkw", "wksokwowkwowk", "wkwww", "wkkwkw", "wlwkwl", "akwkw",
    "wkakaka", "akowkwokwok", "wkakak", "wkw", "awowkwowok", "kwkakakak",
    "awkawkawkkk", "wk", "wkkakak", "nya"
}

# Emoji regex
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

# Preprocessing functions
def replace_text(text: str, old: str, new: str) -> str:
    return text.replace(old, new)

def case_folding(text: str) -> str:
    return text.lower()

def remove_username(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def remove_hashtag(text: str) -> str:
    return re.sub(r'#\w+', '', text)

def remove_retweet(text: str) -> str:
    return re.sub(r'RT\s+', '', text)

def remove_url(text: str) -> str:
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub('', text)

def remove_non_ascii(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', text)

def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)

def remove_punctuation(text: str) -> str:
    return re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_{|}~]', ' ', text)

def remove_extra_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def normalize_words(text: str) -> str:
    return ' '.join(SLANG_DICT.get(w, w) for w in text.split())

def remove_stopwords(text: str) -> str:
    return ' '.join(w for w in text.split() if w.lower() not in STOPWORDS)

def tokenize_text(text: str) -> list[str]:
    return text.split()

def preprocess_text(text: str) -> list[str]:
    text = replace_text(text, "\\\\r", " ")
    text = replace_text(text, "\n", " ")
    text = replace_text(text, "&amp;", " ")
    for fn in (remove_username, remove_hashtag, remove_retweet,
               remove_url, remove_emoji, remove_non_ascii,
               remove_numbers, remove_extra_spaces):
        text = fn(text)
    text = unescape(text)
    text = case_folding(text)
    text = remove_punctuation(text)
    text = normalize_words(text)
    text = remove_stopwords(text)
    return tokenize_text(text)

def vectorize_tokens(tokens: list[str], ft_model: FastText) -> np.ndarray:
    size = ft_model.vector_size
    vecs = [
        ft_model.wv[t] if t in ft_model.wv else np.zeros(size)
        for t in tokens
    ]
    return np.mean(vecs, axis=0) if vecs else np.zeros(size)
