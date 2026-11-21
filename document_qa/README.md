# Product document Q&A

A standalone extension to the CEU beauty assistant. It retrieves product-manual passages from digital and scanned PDFs and returns an answer with page citations. The demo uses fictional products and generates its own PDFs.

## Run locally

Use Python 3.11+ and install the Tesseract command-line tool with its English language data. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python evaluate.py demo_docs
```

Chroma downloads its default local embedding model on first use. The retrieval comparison needs no API key. To generate answers, set the OpenAI API key in your shell and run:

```sh
python rag.py demo_docs "What is the Velora refill code?"
python evaluate.py demo_docs --model gpt-4o-mini
```

The answer commands send retrieved excerpts to OpenAI. Use the included fixtures for public demonstrations.

## What to review

The benchmark uses the same questions with text extraction alone and with OCR enabled. It records whether the expected page is present in the top two passages, and optionally checks the draft answer for the expected fact, citation, or abstention. Read the quoted passages when judging grounding. Matching a quote and a keyword is a smoke test and cannot establish that every assertion is supported.

One case to inspect is the Velora refill code. Its manual is image-only, so direct text extraction cannot index the supporting page. Compare the recorded sources with OCR enabled and inspect the returned quote before judging the answer. The ambiguous refill-code question probes whether the model asks for usable evidence by abstaining when the product is unspecified.

The OCR fallback runs on pages without extracted text. Mixed pages with a selectable heading and a scanned body can still miss facts. OCR output also needs review when characters in product codes look similar.
