"""Write per-question retrieval traces; optionally collect live answers for review."""
import argparse
import json
from pathlib import Path

from scoring import check_answer
from answer import generate
from fixtures import DOCS, HERE
from retrieval import Index, read_pages


def evaluate(live):
    cases = json.loads((HERE / "cases.json").read_text())
    report = {}
    for name, ocr in [("digital_only", False), ("ocr_fallback", True)]:
        index = Index(read_pages(DOCS, ocr))
        rows = []
        for case in cases:
            hits = index.search(case["q"])
            row = {"id": case["id"], "question": case["q"],
                   "retrieved": [{"cite": p.cite, "text": p.text} for p in hits]}
            if "page" in case:
                row["page_hit"] = any(p.cite == case["page"] for p in hits)
                row["evidence_hit"] = any(p.cite == case["page"] and
                                          case["fact"].casefold() in p.text.casefold() for p in hits)
            if live:
                row["answer"] = generate(case["q"], hits)
                row["auto_pass"] = check_answer(case, row["answer"], row.get("evidence_hit"))
            rows.append(row)
        report[name] = rows
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.live:
        from dotenv import load_dotenv
        load_dotenv()
    report = evaluate(args.live)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in report.items():
        scored = [r for r in rows if "page_hit" in r]
        for metric in ["page_hit", "evidence_hit"]:
            print(name, metric, sum(r[metric] for r in scored), "/", len(scored))
        if args.live:
            print(name, "answer checks", sum(r["auto_pass"] for r in rows), "/", len(rows))
