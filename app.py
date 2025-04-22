from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
import gdown
import os
import json

# Import Preprocessing Function
from preprocessing import preprocess_text

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load model GRU dan FastText
GRU_PATH = "gru_model.h5"
if not os.path.exists(GRU_PATH):
    gdown.download("https://drive.google.com/uc?export=download&id=1ZEcrly6zrFSIoeJjKvTYe1V1Qt8NM6P4")
gru_model = tf.keras.models.load_model(GRU_PATH)

FT_BIN = "fasttext.bin"
if not os.path.exists(FT_BIN):
    gdown.download("https://drive.google.com/uc?export=download&id=1nBkDsVjjR-CyYjTBLZ_xgA7GduY6GNaJ", FT_BIN, quiet=False)

FT_NPY = "fasttext.bin.wv.vectors_ngrams.npy"
if not os.path.exists(FT_BIN):
    gdown.download("https://drive.google.com/uc?export=download&id=1POZVZ6gyf8b_KezEBPom_DzQX8VvsCZC", FT_NPY, quiet=False)

ft_model = FastText.load(FT_BIN)

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer_config = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_config))

# Pydantic model
class TextInput(BaseModel):
    text: str

# API Endpoints
@app.get("/preprocess/")
def preprocess_endpoint(text: str):
    tokens = preprocess_text(text)
    return {"cleaned_tokens": tokens}

@app.post("/identification/")
def identification_endpoint(data: TextInput):
    try:
        tokens = preprocess_text(data.text)
        sequence = tokenizer.texts_to_sequences([" ".join(tokens)])
        padded = pad_sequences(
            sequence,
            maxlen=50,
            padding="post",
            truncating="post",
        )
        prediction = gru_model.predict(padded)[0][0]
        label = "Provokasi" if prediction >= 0.5 else "Non-Provokasi"
        return {
            "input": data.text,
            "preprocessed": tokens,
            "probability": float(prediction),
            "classification": label,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
