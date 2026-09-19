import sys
sys.path.insert(0, "src")
import os
os.environ["AIGHT_OFFLINE"] = "1"
from aight_rag.pipeline import build_index, query
from aight_rag.retrieval import assemble_context, simple_rerank
from aight_rag.llm import build_prompt, generate

q = sys.argv[1] if len(sys.argv) > 1 else "what is RAG?"
idx, emb = build_index()
hits = query(idx, emb, q, top_k=6)
hits = simple_rerank(q, hits, top_k=4)
ctx = assemble_context(hits)
prompt = build_prompt(q, hits)
print(ctx)
print("\n---ANSWER---\n")
print(generate(prompt))
