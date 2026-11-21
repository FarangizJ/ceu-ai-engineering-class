"""Page-level ingestion and reproducible local vector embeddings for label lookup."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import fitz
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Passage:
    file: str
    page: int
    method: str
    text: str

    @property
    def cite(self):
        return f"[{self.file} p.{self.page}]"


def read_pages(folder: Path, ocr: bool):
    pages = []
    for path in sorted(folder.glob("*.pdf")):
        with fitz.open(path) as pdf:
            for number, page in enumerate(pdf, 1):
                text = page.get_text().strip()
                method = "digital"
                if not text and ocr:
                    image = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    result = subprocess.run(
                        ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6"],
                        input=image.tobytes("png"), capture_output=True, check=True, timeout=45,
                    )
                    text = result.stdout.decode().strip()
                    method = "ocr"
                if text:
                    pages.append(Passage(path.name, number, method, " ".join(text.split())))
    if not pages:
        raise ValueError("No readable pages found. Generate the samples or check your PDF folder.")
    return pages


class Index:
    def __init__(self, pages):
        self.pages = pages
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vectors = self.vectorizer.fit_transform(p.text for p in pages)
        import chromadb
        from uuid import uuid4
        self.collection = chromadb.EphemeralClient().create_collection(
            "labels-" + uuid4().hex, metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.collection.add(
            ids=[str(i) for i in range(len(pages))],
            documents=[p.text for p in pages],
            metadatas=[{"file": p.file, "page": p.page, "method": p.method} for p in pages],
            embeddings=self.vectors.toarray().tolist(),
        )

    def search(self, question, k=2):
        vector = self.vectorizer.transform([question])
        if not vector.nnz:
            return []
        result = self.collection.query(query_embeddings=vector.toarray().tolist(),
                                       n_results=min(k, len(self.pages)),
                                       include=["documents", "metadatas", "distances"])
        return [Passage(m["file"], m["page"], m["method"], doc)
                for m, doc, distance in zip(result["metadatas"][0], result["documents"][0],
                                             result["distances"][0]) if distance < 0.999999]
