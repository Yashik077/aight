"""Fixed-size chunking. Probably terrible but let's start."""


def fixed_chunk(text, chunk_size=500):
    # v1: dumb slicing, no overlap, splits mid-word lol
    chunks = []
    for i in range(0, len(text), chunk_size):
        c = text[i:i+chunk_size]
        if c.strip():
            chunks.append(c)
    return chunks


def chunk_documents(docs, chunk_size=500):
    out = []
    for d in docs:
        for idx, ch in enumerate(fixed_chunk(d["text"], chunk_size)):
            out.append({"source": d["source"], "chunk_id": idx, "text": ch})
    return out
