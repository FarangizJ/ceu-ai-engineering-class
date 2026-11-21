import argparse

from answer import generate
from fixtures import DOCS
from retrieval import Index, read_pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()
    hits = Index(read_pages(DOCS, True)).search(args.question)
    for passage in hits:
        print(passage.cite, passage.method, passage.text)
    if not args.evidence_only:
        from dotenv import load_dotenv
        load_dotenv()
        print(generate(args.question, hits))
