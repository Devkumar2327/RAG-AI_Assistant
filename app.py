

import os
import streamlit as st
from google import genai

from vector_store import build_index
from rag import answer_query, TOP_K

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🏠", layout="centered")

st.title("🏠 IITB Insti-Assist")
st.caption("Hostel & Campus Life Assistant — grounded answers about IIT Bombay hostel fees, rules, and campus housing, built with RAG.")

# --- Sidebar: API key + info ---
with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get one free at aistudio.google.com/app/apikey. Not stored anywhere — only used for this session.",
    )
    st.markdown("---")
    st.header("Knowledge base")
    st.markdown(
        "- Hostel Fee Payment Circular (Autumn 2026-27)\n"
        "- Hostel Fee Structure tables (UG/PG, new & continuing)\n"
        "- New Entrants Hostel General Instructions\n"
        "- Hostel 10 Rules & Fines (representative hostel rules)\n"
        "- Campus Housing & Dining Facilities Overview"
    )
    st.markdown("---")
    st.caption("This assistant only answers from the 5 documents above. If your question is out of scope, it will say so rather than guess.")

# --- Build / load vector index (cached across reruns) ---
@st.cache_resource(show_spinner="Building knowledge base index...")
def get_index():
    return build_index()

index, chunks = get_index()

# --- Chat state ---
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("Sources used"):
                for s in turn["sources"]:
                    st.markdown(f"- `{s}`")
        if turn["role"] == "assistant" and turn.get("retrieved"):
            with st.expander("Retrieved chunks (debug view)"):
                for r in turn["retrieved"]:
                    st.markdown(f"**{r['source']}** (score: {r['score']:.3f})")
                    st.text(r["text"][:400])

query = st.chat_input("Ask about IIT Bombay hostel fees, rules, or campus housing...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not api_key_input:
            msg = "Please enter your Gemini API key in the sidebar to get an answer."
            st.warning(msg)
            st.session_state.history.append({"role": "assistant", "content": msg})
        else:
            with st.spinner("Retrieving relevant documents and generating answer..."):
                client = genai.Client(api_key=api_key_input)
                result = answer_query(query, client=client, index=index, chunks=chunks, k=TOP_K)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources used"):
                    for s in result["sources"]:
                        st.markdown(f"- `{s}`")
            with st.expander("Retrieved chunks (debug view)"):
                for r in result["retrieved"]:
                    st.markdown(f"**{r['source']}** (score: {r['score']:.3f})")
                    st.text(r["text"][:400])
            st.session_state.history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "retrieved": result["retrieved"],
            })

st.markdown("---")
st.caption("Example questions: \"When is the last date to pay hostel fees for Autumn 2026-27?\" · \"What's the fine for an unbooked overnight guest?\" · \"How much is the mess advance for a new PG student?\"")
