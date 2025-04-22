# FastAPI Doxing Provocation Identification API

## 📦 Tech Stack & Dependencies

- **Python 3.10+**  
- [FastAPI](https://fastapi.tiangolo.com/)  
- [Uvicorn](https://www.uvicorn.org/)  
- [TensorFlow](https://www.tensorflow.org/) (GRU, eager execution)  
- [Gensim](https://radimrehurek.com/gensim/) (FastText)  
- [Sastrawi](https://github.com/har07/PySastrawi) (Indonesian stopwords)  
- [gdown](https://github.com/wkentaro/gdown) (download from Google Drive)  
- **numpy**, **pydantic**

All requirements are pinned in `requirements.txt`.

---

## 📥 Installation

```bash
python -m venv .venv
.venv\Scripts\activate 
pip install --upgrade pip
pip install -r requirements.txt

```
## ▶️ Running the Server
```bash
uvicorn app:app --reload
```


