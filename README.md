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

## What's in here
iitb-insti-assist/
├── data/ 5 source docs, cleaned to plain text
├── src/
│ ├── ingest.py chunking
│ ├── vector_store.py embedding + FAISS
│ ├── rag.py retrieval + grounded prompt + Gemini call
│ └── app.py the Streamlit UI
├── index/ cached index (auto-generated, ignore this)
└── WRITEUP.md the full write-up — scope, sourcing, chunking rationale, limitations

## Testing without the UI

```bash
python3 src/ingest.py        # see how documents got chunked
python3 src/vector_store.py  # build index, sanity-check retrieval — no API key needed
GEMINI_API_KEY=AIza... python3 src/rag.py   # full round-trip
```

## Try asking it

- "When is the last date to pay hostel fees for Autumn 2026-27?"
- "What's the fine for paying late?"
- "Total fee for a new PG student joining in 2026-27?"
- "Fine for an unbooked overnight guest?"
- "What facilities does every hostel have?"
- "What's the capital of France?" — should get an honest "I don't know"

## One thing worth knowing

Everything runs locally except the final answer-generation step, which hits 
the Gemini API. Embedding and retrieval never leave your machine.

Curious about the *why* behind the chunking strategy or the known gaps? 
That's all in `WRITEUP.md`.