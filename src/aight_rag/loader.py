"""Doc loader v2: txt/md/pdf + cleanup.
Dead-end: PyPDF2 gave mojibake, switched to pypdf. Dehyphenate + normalize whitespace.
"""
from pathlib import Path
import re


def clean_text(t):
    t = t.replace("-\n", "")  # dehyphenate
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_documents(folder):
    docs = []
    folder = Path(folder)
    if not folder.exists():
        return docs
    for f in sorted(folder.glob("*")):
        try:
            if f.suffix in [".txt", ".md"]:
                text = f.read_text(encoding="utf-8", errors="ignore")
            elif f.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader
                except ImportError:
                    print("pypdf not installed, skipping pdf", f)
                    continue
                reader = PdfReader(str(f))
                text = "\n".join([(p.extract_text() or "") for p in reader.pages])
            else:
                continue
            text = clean_text(text)
            if text:
                docs.append({"source": str(f.name), "text": text})
        except Exception as e:
            print(f"skip {f}: {e}")
    return docs


if __name__ == "__main__":
    print(load_documents("data/sample_docs"))
