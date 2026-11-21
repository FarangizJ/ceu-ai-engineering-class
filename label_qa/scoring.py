"""Conservative automatic checks; human reviewers still check every claim."""
import re


def check_answer(case, answer, evidence_hit=None):
    lower = answer.lower()
    if case.get("expect") == "clarify which product":
        return "?" in answer and any(t in lower for t in ("which", "mean", "clarify", "specify"))
    if case.get("expect") == "decline" or not evidence_hit:
        return lower.strip().rstrip(".") == "insufficient evidence"
    return (re.search(r"(?<!\w)" + re.escape(case["fact"].casefold()) + r"(?!\w)",
                      answer.casefold()) is not None and case["page"] in answer)
