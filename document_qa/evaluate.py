"""Compare searchable PDF text alone against PDF text with OCR fallback."""
import argparse
import json
from pathlib import Path

from openai import OpenAI

from rag import answer, build_index, retrieve

CASES = [
    ("What size is the Lumina cleansing gel?", ("lumina.pdf", 1), "150"),
    ("What is the Lumina refill code?", ("lumina.pdf", 1), "LUM-42"),
    ("What color is the Lumina bottle?", ("lumina.pdf", 2), "amber"),
    ("What is Lumina's outer carton made of?", ("lumina.pdf", 2), "kraft"),
    ("What size is the Velora hand cream?", ("velora.pdf", 1), "75"),
    ("What is the Velora refill code?", ("velora.pdf", 1), "VEL-19"),
    ("What color is the Velora tube?", ("velora.pdf", 2), "blue"),
    ("What is Velora's outer carton made of?", ("velora.pdf", 2), "white"),
    ("What is the wholesale price of Lumina?", None, None),
    ("What is the refill code?", None, None),
    ("What is the shelf life of Velora?", None, None),
]


def evaluate(index, client, model):
    rows = []
    for question, expected, keyword in CASES:
        passages = retrieve(index, question, top_k=2)
        sources = {(item["file"], item["page"]) for item in passages}
        row = {
            "question": question,
            "expected_source": expected,
            "retrieved_sources": sorted(sources),
            "retrieval_hit": expected in sources if expected else None,
        }
        if client is not None:
            result = answer(client, model, question, passages)
            cited = {(item["file"], item["page"]) for item in result["citations"]}
            row["output"] = result
            row["answer_check"] = (
                expected in cited and keyword.casefold() in result["answer"].casefold()
                if expected else not result["answer"]
            )
        rows.append(row)
    return {
        "retrieval_hits": sum(row["retrieval_hit"] is True for row in rows),
        "answerable_cases": sum(bool(row["expected_source"]) for row in rows),
        "automatic_answer_checks_passed": (
            sum(row["answer_check"] for row in rows) if client is not None else None
        ),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model", help="Run live answer checks with this OpenAI model")
    args = parser.parse_args()
    client = OpenAI() if args.model else None
    report = {}
    for label, use_ocr in [("text_only", False), ("ocr_fallback", True)]:
        index = build_index(args.folder, ocr=use_ocr)
        report[label] = evaluate(index, client, args.model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
