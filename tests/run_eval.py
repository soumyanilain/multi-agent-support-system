"""
Evaluation harness for the Specialist Agent's RAG pipeline.

Produces two tables for the project report:

  1. Classification accuracy across the provided test cases.
  2. A like-for-like comparison of dense-only retrieval against
     hybrid fusion retrieval (BM25 + dense, merged with RRF).

Run from the project root, with the virtual environment active:

    python -m tests.run_eval

This talks to the pipeline directly rather than through the A2A
server, so the Specialist Agent does not need to be running.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

load_dotenv()

from specialist_agent.rag_pipeline import get_pipeline  # noqa: E402


TEST_CASES_PATH = BASE_DIR / "tests" / "test_cases.json"

# Categories the mock support form's dropdown can accept. The
# knowledge base also declares Email and Security, which the form
# cannot take -- those are expected to fail as unsupported_category.
FORM_CATEGORIES = {
    "Account Access",
    "Hardware",
    "Software",
    "Network",
}

CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.30")
)

# Deliberately outside the knowledge base, to confirm that
# out-of-scope questions are rejected rather than answered.
OUT_OF_SCOPE_QUESTION = (
    "How do I rebuild the transmission in my car?"
)


def load_test_cases():

    with open(TEST_CASES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def classify(pipeline, question, use_fusion=True):
    """
    Run one question through the pipeline and apply the same
    policies the Specialist Agent applies.

    Returns a dictionary describing the outcome.
    """

    try:
        result = pipeline.answer(
            question,
            use_fusion=use_fusion
        )

    except Exception as error:
        return {
            "outcome": "error",
            "category": None,
            "confidence": 0.0,
            "sources": [],
            "detail": str(error),
        }

    confidence = result.get("confidence", 0.0)
    category = result.get("category", "")

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "outcome": "insufficient_context",
            "category": category,
            "confidence": confidence,
            "sources": result.get("sources", []),
            "detail": "Below relevance threshold.",
        }

    if category not in FORM_CATEGORIES:
        return {
            "outcome": "unsupported_category",
            "category": category,
            "confidence": confidence,
            "sources": result.get("sources", []),
            "detail": "Form cannot accept this category.",
        }

    return {
        "outcome": "completed",
        "category": category,
        "confidence": confidence,
        "sources": result.get("sources", []),
        "detail": "",
    }


def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_accuracy_table(pipeline, test_cases):
    """Table 1: classification accuracy on the provided test cases."""

    print_header("TABLE 1  --  CLASSIFICATION ACCURACY (hybrid fusion)")

    print(
        f"{'#':<3} {'Question':<46} {'Expected':<16} "
        f"{'Predicted':<16} {'Conf':<7} {'Result'}"
    )
    print("-" * 110)

    correct = 0
    submittable = 0

    for case in test_cases:

        question = case["request"]
        expected = case["expected_category"]

        outcome = classify(pipeline, question)

        predicted = outcome["category"] or "-"

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        if outcome["outcome"] == "completed":
            submittable += 1
            verdict = "OK" if is_correct else "WRONG"
        else:
            # A correct category the form cannot accept is still a
            # correct classification -- it is a form limitation.
            verdict = outcome["outcome"]

        print(
            f"{case['id']:<3} {question[:45]:<46} {expected:<16} "
            f"{predicted:<16} {outcome['confidence']:<7} {verdict}"
        )

    total = len(test_cases)

    print("-" * 110)
    print(
        f"Classification accuracy: {correct}/{total} "
        f"({100 * correct / total:.0f}%)"
    )
    print(
        f"Submittable to the support form: {submittable}/{total} "
        "(the remainder are valid categories the form cannot accept)"
    )


def run_failure_table(pipeline):
    """Table 2: the failure modes the system is expected to detect."""

    print_header("TABLE 2  --  FAILURE MODE DETECTION")

    outcome = classify(pipeline, OUT_OF_SCOPE_QUESTION)

    print(f"Question:   {OUT_OF_SCOPE_QUESTION}")
    print(f"Outcome:    {outcome['outcome']}")
    print(f"Confidence: {outcome['confidence']}")
    print(f"Threshold:  {CONFIDENCE_THRESHOLD}")

    if outcome["outcome"] == "insufficient_context":
        print("\nPASS -- out-of-scope question was rejected.")
    else:
        print(
            "\nFAIL -- an out-of-scope question was answered. "
            "Consider raising CONFIDENCE_THRESHOLD."
        )


def run_retrieval_comparison(pipeline, test_cases):
    """
    Table 3: dense-only retrieval vs hybrid fusion retrieval.

    Both strategies run the same multi-query expansion, so the only
    difference is whether BM25 rankings are fused into the result.
    """

    print_header(
        "TABLE 3  --  DENSE-ONLY vs HYBRID FUSION RETRIEVAL"
    )

    print(
        f"{'#':<3} {'Question':<40} "
        f"{'Dense-only sources':<34} {'Fusion sources'}"
    )
    print("-" * 118)

    questions = [case["request"] for case in test_cases]
    questions.append(OUT_OF_SCOPE_QUESTION)

    changed = 0

    for position, question in enumerate(questions, start=1):

        dense = pipeline.retrieve_multi_query(
            question,
            use_fusion=False
        )

        fusion = pipeline.retrieve_multi_query(
            question,
            use_fusion=True
        )

        dense_sources = sorted(
            {item["source"] for item in dense}
        )

        fusion_sources = sorted(
            {item["source"] for item in fusion}
        )

        if dense_sources != fusion_sources:
            changed += 1
            marker = "  <-- differs"
        else:
            marker = ""

        print(
            f"{position:<3} {question[:39]:<40} "
            f"{', '.join(dense_sources)[:33]:<34} "
            f"{', '.join(fusion_sources)[:40]}{marker}"
        )

    print("-" * 118)
    print(
        f"Retrieved document set changed on {changed}/{len(questions)} "
        "questions."
    )
    print(
        "\nNote: on a small, cleanly separated knowledge base, dense\n"
        "retrieval already performs well, so fusion is expected to\n"
        "change which passages are retrieved more than which category\n"
        "is finally assigned."
    )


def main():

    print("Building RAG pipeline (this takes a few seconds)...")

    pipeline = get_pipeline()

    test_cases = load_test_cases()

    run_accuracy_table(pipeline, test_cases)

    run_failure_table(pipeline)

    run_retrieval_comparison(pipeline, test_cases)

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()