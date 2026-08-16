# IITB Insti-Assist

Ask it about hostel fees, hostel rules, or campus housing — it answers from 
real IITB documents, and says "I don't know" instead of making things up when 
a question falls outside what it actually knows.

## What it does

Five hostel/campus documents get chunked, embedded, and indexed. When you 
ask something, it retrieves the most relevant chunks, hands them to Gemini 
along with a prompt that forbids answering from anything else, and shows you 
which source it pulled from.

docs → chunk (ingest.py) → embed (vector_store.py) → FAISS index
│
Streamlit UI ← Gemini ← retrieval + grounded prompt (rag.py)

## Getting it running

**1. Install dependencies** (Python 3.10+)
```bash
pip install -r requirements.txt
```
First run pulls the `all-MiniLM-L6-v2` embedding model (~90MB) from Hugging Face.

**2. Grab a free Gemini key** from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Either export it before launching:
```bash
export GEMINI_API_KEY=AIza...
```
or just paste it into the sidebar once the app's open (nothing gets stored).

**3. Launch**
```bash
streamlit run src/app.py
```
First launch builds the FAISS index and caches it — every launch after that's instant.
