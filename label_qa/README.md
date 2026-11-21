# Can a beauty assistant cite the label?

A product-label Q&A example for the CEU beauty chatbot project. The two fictional PDFs let anyone inspect the source, rerun retrieval without an API key, and see exactly what the model receives. The original Chainlit app and this CLI have separate entry points.

## What did OCR change?

Ten questions cover seven facts printed on the labels, one ambiguous request and two facts the labels never state. The experiment changes only whether image-only pages receive English OCR. Both runs use the same PDFs, questions, word unigram/bigram TF-IDF embeddings, cosine search in Chroma, and two retrieved passages. Each short page forms one chunk. TF-IDF keeps the small experiment local and inspectable. Paraphrases without shared vocabulary can still miss.

| Measured on seven factual questions | Digital extraction | Digital + OCR |
|---|---:|---:|
| Correct PDF and page among the top two | 4/7 | 7/7 |
| Correct page also contains the expected answer text | 4/7 | 6/7 |

The two product cards occupy [catalog.pdf](samples/catalog.pdf). The packaging insert is a [raster-only scan](samples/insert.pdf). [Read the ten questions and gold labels](cases.json) or [inspect every retrieved passage in both runs](results.json).

**Failure S1.** Asked how many refill trays come together, digital extraction returns the Cloud Balm product card. It identifies the right product but says nothing about tray counts. OCR recovers “Cloud Balm refill trays are packed in sets of four.” A supported answer is “Four trays [insert.pdf p.1].”

**Failure S3.** The fixture prints a faint batch code, LUMI-81. Tesseract recovers the mailing instructions above it and drops the entire code line. S3 passes the page check and fails the evidence check. A model reading that passage should decline. The [fixture generator](fixtures.py) preserves the original text for comparison.

## Run it

Use Python 3.11 and install Tesseract with English language data (Ubuntu packages `tesseract-ocr` and `tesseract-ocr-eng`, or `brew install tesseract` on macOS). From the repository root:

```sh
cd label_qa
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python ask.py 'How many Cloud Balm refill trays are packed together?' --evidence-only
python evaluate.py --out my-results.json
python -m unittest test_scoring
```

The PDFs are included. To recreate them, run `python fixtures.py`. Recorded versions were Tesseract 5.5.0, PyMuPDF 1.26.6, scikit-learn 1.8.0 and Chroma 1.5.9. Other OCR versions can read the faded line differently.

For OpenAI generation, copy `.env.example` to `.env` in this directory and enter your API key. The second command below makes 20 model calls.

```sh
python ask.py 'What size does the product card list?'
python evaluate.py --live --out my-answers.json
```

## How to review the answers

These examples are reference answers. The published run measures retrieval. Live model outputs and answer-level scores require an API key.

| Case | Expected behavior | Example response |
|---|---|---|
| D2, material of the Reed Mist bottle | State the material and cite the card | Glass [catalog.pdf p.2]. |
| A1, “What size does the product card list?” | Both cards contain sizes. Ask which product the reader means | Do you mean Cloud Balm or Reed Mist? |
| U1, retail price of Cloud Balm | Decline because no price appears | Insufficient evidence. |

The automatic answer check requires the gold fact and exact page citation for a supported fact, a clarification question for A1, and an explicit abstention when the retrieved evidence is missing. Review all ten responses against the passages as well. Mark each pass only if every factual claim has support, the citation identifies its page, and the response answers, clarifies or declines appropriately. A response can match the automatic rule and still introduce an unsupported detail. Record reviewed passes out of ten separately for each run. This small, authored English fixture measures these cases only.
