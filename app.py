from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json, re
from html import unescape
import tensorflow as tf
import gdown
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
tf.config.run_functions_eagerly(True)

# Inisialisasi aplikasi FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load GRU Model
url = "https://drive.google.com/uc?export=download&id=1fso1UZPipDeqsVUpY52t4wffDYtWsrpE"
model_path = "gru_model.h5"
try:
    import os
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
except Exception as e:
    print("Gagal mendownload model:", e)
gru_model = tf.keras.models.load_model(model_path)

# Load Model Fasttext
fasttext_bin_url = "https://drive.google.com/uc?export=download&id=1ZKz-LCWB_MQW-2_tmpR9xpBX8FnRZdLK"
fasttext_npy_url = "https://drive.google.com/uc?export=download&id=1Hgqr2Jvxu-4UtdnOC5AvTDYSRtWyLjMk"
bin_path = "fasttext.bin"
npy_path = "fasttext.bin.wv.vectors_ngrams.npy"

if not os.path.exists(bin_path):
    print("Mengunduh fasttext.bin...")
    gdown.download(fasttext_bin_url, bin_path, quiet=False)

if not os.path.exists(npy_path):
    print("Mengunduh fasttext.bin.wv.vectors_ngrams.npy...")
    gdown.download(fasttext_npy_url, npy_path, quiet=False)
print("Memuat model FastText...")
ft_model = FastText.load(bin_path)

# Load slang dictionary
with open("final_slang.txt", "r", encoding="utf-8") as f:
    slang_dict = json.load(f)

# Load stopwords
factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())
custom_stopwords = {
    "huhu", "wkwk", "dong", "akwkakakak", "hehe", "akxkakskak", "sikxskxk",
    "wkwkwkwk", "wkwkwk", "wkakka", "wkkw", "wkwkw", "wkwkwkwk", "wkwkwkwkwk",
    "hehehe", "heheh", "dongg", "wkwwkwk", "wkwkwkkw", "wkwkwkw", "awokwkwkwkw",
    "aowkwoowkwwkwkkw", "wkwkwwk", "wkwkkww", "wkwkk", "wkwkwkk", "hehehaha",
    "ehehehe", "mwehehe", "hihi", "wkwkak", "wkwkwkkwk", "wkwkwkwkwkwkwk",
    "kwkwkw", "wkwkkw", "wksokwowkwowk", "wkwww", "wkkwkw", "wlwkwl", "akwkw",
    "wkakaka", "akowkwokwok", "wkakak", "wkw", "awowkwowok", "kwkakakak",
    "awkawkawkkk", "wk", "wkkakak", "nya"
}
stopwords.update(custom_stopwords)

emoji_pattern = re.compile(
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
    "]+", flags=re.UNICODE
)

# Fungsi-fungsi preprocessing
def REPLACE(text, old, new):
    return text.replace(old, new)

def CASE_FOLDING(text):
    return text.lower()

def REMOVE_USERNAME(text):
    return re.sub(r'@\w+', '', text)

def REMOVE_HASHTAG(text):
    return re.sub(r'#\w+', '', text)

def REMOVE_RETWEET(text):
    return re.sub(r'RT\s+', '', text)

def REMOVE_URL(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def REMOVE_EMOJI(text):
    return emoji_pattern.sub(r'', text)

def REMOVE_NON_ASCII(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def REMOVE_NUMBERS(text):
    return re.sub(r'\d+', '', text)

def REMOVE_EXTRA_SPACES(text):
    return re.sub(r'\s+', ' ', text).strip()

def REMOVE_PUNCTUATION(text):
    return re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_{|}~]', ' ', text)

def NORMALIZE_WORDS(text):
    words = text.split()
    normalized_words = [slang_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)

def REMOVE_STOPWORDS(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

def TOKENIZE_TEXT(text):
    return text.split()

def preprocess_text(text: str) -> list:
    text = REPLACE(text, "\\\\r", " ")
    text = REPLACE(text, "\n", " ")
    text = REPLACE(text, "&amp;", " ")
    text = REMOVE_USERNAME(text)
    text = REMOVE_HASHTAG(text)
    text = REMOVE_RETWEET(text)
    text = REMOVE_URL(text)
    text = REMOVE_EMOJI(text)
    text = REMOVE_NON_ASCII(text)
    text = REMOVE_NUMBERS(text)
    text = REMOVE_EXTRA_SPACES(text)
    text = unescape(text)
    text = CASE_FOLDING(text)
    text = REMOVE_PUNCTUATION(text)
    text = NORMALIZE_WORDS(text)
    text = REMOVE_STOPWORDS(text)
    tokens = TOKENIZE_TEXT(text)
    return tokens


# Load Tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer_config = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_config))
max_length = 50
vector_size = ft_model.vector_size 

def vectorize_tokens(tokens: list) -> np.ndarray:
    vectors = []
    for token in tokens:
        if token in ft_model.wv.key_to_index:
            vectors.append(ft_model.wv[token])
        else:
            vectors.append(np.zeros(vector_size))
    if len(vectors) == 0:
        return np.zeros(vector_size)
    else:
        return np.mean(vectors, axis=0)

class TextInput(BaseModel):
    text: str

# Defaul Endpoint
@app.get("/")
def index():
    return {"message": "Hello World"}

# Endpoint /preprocess/
@app.get("/preprocess/")
def preprocess_endpoint(text: str):
    tokens = preprocess_text(text)
    return {"cleaned_tokens": tokens}

# Endpoint /identification/
@app.post("/identification/")
def identification_endpoint(data: TextInput):
    try:
        tokens = preprocess_text(data.text)
        sequences = tokenizer.texts_to_sequences([" ".join(tokens)])
        padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        model_existing = tf.keras.models.load_model("gru_model.h5")
        prediction = model_existing.predict(padded_sequence)[0][0]
        label = "Provokasi" if prediction >= 0.5 else "Non-Provokasi"
        return {"input": data.text, "preprocessed": tokens, "probability": float(prediction), "classification": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
