"""Naive doc loader - v1. Just txt/md, no pdf yet."""
import os
from pathlib import Path


def load_documents(folder):
    docs = []
    folder = Path(folder)
    if not folder.exists():
        return docs
    for f in folder.glob("*"):
        if f.suffix in [".txt", ".md"]:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"skip {f}: {e}")
                continue
            docs.append({"source": str(f.name), "text": text})
    return docs


if __name__ == "__main__":
    print(load_documents("data/sample_docs"))
