
import os
from google import genai
from google.genai import types

from vector_store import search, build_index

MODEL = "gemini-3.5-flash"
TOP_K = 4
SIMILARITY_FLOOR = 0.20  # below this, we treat retrieval as "nothing relevant found"

SYSTEM_PROMPT = """You are IITB Insti-Assist, an assistant that answers questions about \
IIT Bombay hostel fees, hostel rules, and campus housing/dining life.

STRICT GROUNDING RULES:
1. Answer ONLY using the information in the "CONTEXT" block below. Do not use outside \
knowledge about IIT Bombay or general assumptions about how hostels usually work.
2. If the context does not contain enough information to answer the question, say clearly: \
"I don't know based on the documents I have — you may want to check with the Hostel \
Coordinating Unit (HCU) or Dean of Student Affairs directly." Do not guess or fill gaps.
3. When you do answer, be specific and cite figures/dates/rules exactly as given in the \
context (do not round or approximate numbers).
4. After your answer, list which source document(s) you used, by filename.
5. Keep answers concise and direct — this is a quick-reference assistant, not an essay writer.

CONTEXT:
{context}
"""


def format_context(chunks):
    if not chunks:
        return "(no relevant context retrieved)"
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_query(query: str, client: genai.Client = None, index=None, chunks=None, k: int = TOP_K):
    """
    Run the full RAG pipeline for one query.
    Returns dict: {"answer": str, "sources": list[str], "retrieved": list[dict], "grounded": bool}
    """
    results = search(query, k=k, index=index, chunks=chunks)
    grounded = bool(results) and results[0]["score"] >= SIMILARITY_FLOOR

    if not grounded:
        return {
            "answer": (
                "I don't know based on the documents I have — this looks outside my "
                "current knowledge base (Hostel & Campus Life at IIT Bombay: fees, "
                "hostel rules, and housing/dining facilities). You may want to check "
                "with the Hostel Coordinating Unit (HCU) or Dean of Student Affairs directly."
            ),
            "sources": [],
            "retrieved": results,
            "grounded": False,
        }

    context = format_context(results)
    system = SYSTEM_PROMPT.format(context=context)

    if client is None:
        client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from env

    response = client.models.generate_content(
        model=MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=600,
        ),
    )
    answer_text = response.text

    sources = sorted(set(c["source"] for c in results))
    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved": results,
        "grounded": True,
    }


if __name__ == "__main__":
    index, chunks = build_index()
    q = "What is the fine for keeping an unbooked guest overnight in Hostel 10?"
    print("Query:", q)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("(Set GEMINI_API_KEY to actually call the model — showing retrieval only)")
        for r in search(q, k=TOP_K, index=index, chunks=chunks):
            print(f"  [{r['score']:.3f}] {r['source']}")
    else:
        result = answer_query(q, index=index, chunks=chunks)
        print(result["answer"])
        print("Sources:", result["sources"])
