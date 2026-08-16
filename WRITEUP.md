# IITB Insti-Assist — Project Write-up

## Why this scope
I picked Hostel & Campus Life over Academic, Council/Club, or General Insti Assistant. 
Hostel info is where a generic LLM is most likely to confidently guess wrong — fee 
deadlines, fines, due dates are exact numbers tied to a specific year's circular. It's 
also a scope with real, well-structured source docs available (fee circulars, rule 
sheets), making it doable as a RAG project in a week without sprawling into academics.

## Data sources (5 docs, 3 sub-areas)
1. Hostel Fee Payment Circular (Autumn 2026-27) — payment schedule + late fines
2. Hostel Fee Structure tables — fee breakdown by batch/programme
3. New Entrants' Hostel Instructions — onboarding, room allotment, mess sign-up
4. Hostel 10 Rules & Fines — one hostel's rulebook, flagged as representative 
   (each of IITB's 17 hostels has its own version)
5. Campus Housing & Dining Overview — shared facilities, vacation room-retention rules

First two were PDFs I already had (most current). Rest were pulled via web search 
of iitb.ac.in / gymkhana.iitb.ac.in / dosa.iitb.ac.in, cleaned from PDF/HTML to plain text.

## Chunking approach
- Split on paragraph boundaries first, then packed into ~900-char chunks, 150-char overlap
- Why: these docs are lists of discrete facts (a rule + its fine, a fee row) — 
  naive fixed-size splitting risks separating a rule from its fine amount
- Long paragraphs (e.g. the 13-condition fee note) get a sliding window instead 
  of being force-fit into one chunk
- Overlap ensures boundary facts (e.g. "Fine: Rs.100/day") stay with their context
- Result: 31 chunks from 5 docs — small enough for top-k=4 retrieval to stay precise

## Retrieval, grounding, refusal
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, FAISS flat index, cosine similarity
- Top-4 chunks retrieved per query. If best score < 0.20 → fixed "I don't know", 
  LLM never called — out-of-scope questions can't be hallucinated about
- When context is found: system prompt bans outside knowledge, requires the model 
  to say "I don't know" if context doesn't actually answer, and forces exact 
  figures/dates from source (no approximating)
- Every answer shows its source doc(s) + a "retrieved chunks" debug view

## Limitations / next steps
- **Single hostel's rules** — Hostel 10 stands in for all 17; a production version 
  would ingest all 17 and filter by hostel name
- **Coarse similarity cutoff** — one global threshold (0.20) can misfire on 
  borderline queries; a learned classifier or retrieve-then-verify step would help
- **No conversational memory** — each question is independent; follow-ups lose context
- **Static index** — fees/deadlines change every semester; needs a refresh pipeline
- **No user PDF upload** — letting students upload their own hostel's rule sheet 
  would make this useful across all 17 hostels without me pre-ingesting each one