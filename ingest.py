
import os
import re
import glob

CHUNK_SIZE = 900          # target characters per chunk
CHUNK_OVERLAP = 150       # characters of overlap between consecutive chunks
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_documents(data_dir: str = DATA_DIR):
    """Load all .txt files from data_dir. Returns list of (filename, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((os.path.basename(path), text))
    return docs


def split_into_paragraphs(text: str):
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Greedily pack paragraphs into chunks of ~chunk_size chars, with overlap."""
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current = ""

    for para in paragraphs:
        # If a single paragraph is longer than chunk_size (e.g. a big fee table),
        # split it on its own with a sliding window.
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end].strip())
                start = end - overlap
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current.strip())
            # carry a small overlap from the end of the previous chunk
            tail = current[-overlap:] if current else ""
            current = (tail + "\n\n" + para).strip()

    if current:
        chunks.append(current.strip())

    return chunks


def build_chunks(data_dir: str = DATA_DIR):
    """Load all docs and produce the final list of chunk records."""
    all_chunks = []
    for filename, text in load_documents(data_dir):
        doc_chunks = chunk_text(text)
        for i, chunk in enumerate(doc_chunks):
            all_chunks.append({
                "id": f"{filename}::chunk{i}",
                "source": filename,
                "text": chunk,
            })
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(load_documents())} documents.\n")
    for c in chunks[:3]:
        print(f"[{c['id']}] ({len(c['text'])} chars)")
        print(c["text"][:200].replace("\n", " "))
        print("---")
