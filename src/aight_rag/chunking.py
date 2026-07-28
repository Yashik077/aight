"""Chunking v3: + semantic-ish sentence boundaries.
Naive sentence split on '. ' + greedy pack. Breaks on PDFs (no punctuation), fix next.
"""


def fixed_chunk(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        c = text[i:i+chunk_size]
        if c.strip():
            chunks.append(c)
    return chunks


def recursive_chunk(text, chunk_size=500, overlap=50, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    # try to split by first separator that actually splits
    for sep in separators:
        if sep == "":
            # fallback to fixed with overlap
            out = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                piece = text[start:end]
                if piece.strip():
                    out.append(piece)
                if end >= len(text):
                    break
                start = end - overlap
            return out
        if sep in text:
            parts = text.split(sep)
            chunks, cur = [], ""
            for p in parts:
                cand = (cur + sep + p) if cur else p
                if len(cand) <= chunk_size:
                    cur = cand
                else:
                    if cur:
                        chunks.append(cur.strip())
                        # overlap: keep tail
                        cur = (cur[-overlap:] + sep + p) if overlap else p
                        if len(cur) > chunk_size:
                            # recurse
                            chunks.extend(recursive_chunk(p, chunk_size, overlap, separators[separators.index(sep)+1:]))
                            cur = ""
                    else:
                        chunks.extend(recursive_chunk(p, chunk_size, overlap, separators[separators.index(sep)+1:]))
            if cur.strip():
                chunks.append(cur.strip())
            return [c for c in chunks if c.strip()]
    return [text]


def chunk_documents(docs, chunk_size=500, overlap=50, method="recursive"):
    out = []
    for d in docs:
        if method == "semantic":
            pieces = semantic_chunk(d["text"], chunk_size, overlap)
        elif method == "recursive":
            pieces = recursive_chunk(d["text"], chunk_size, overlap)
        else:
            pieces = fixed_chunk(d["text"], chunk_size)
        for idx, ch in enumerate(pieces):
            out.append({"source": d["source"], "chunk_id": idx, "text": ch})
    return out


def semantic_chunk(text, chunk_size=500, overlap=50):
    import re
    # v1: split sentences, pack greedily. Fails on PDF line-breaks, TODO.
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= chunk_size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
                # overlap by last N chars
                cur = (cur[-overlap:] + " " + s).strip() if overlap else s
            else:
                chunks.append(s[:chunk_size])
                cur = s[chunk_size:]
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
