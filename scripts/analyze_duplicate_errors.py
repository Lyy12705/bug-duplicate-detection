#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector
from duplicate_ticket_detection.splits import train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Top-1 duplicate ranking errors for manual error analysis.")
    parser.add_argument("--tickets", required=True)
    parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    parser.add_argument("--model-dir", help="Saved SBERT model directory. Required for --method sbert")
    parser.add_argument("--combine", default="mean")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", default="reports/top1_error_analysis.csv")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    detector = build_detector(args, split.train)
    rows = collect_top1_errors(detector, split.test, split.train, top_k=args.top_k)
    write_errors_csv(rows, args.output_csv)

    evaluable = count_evaluable_queries(split.test, split.train)
    print(f"train={len(split.train)}")
    print(f"test={len(split.test)}")
    print(f"evaluable_queries={evaluable}")
    print(f"top1_errors={len(rows)}")
    print(f"top1_error_rate={0.0 if evaluable == 0 else len(rows) / evaluable:.4f}")
    print(f"output_csv={args.output_csv}")
    return 0


def build_detector(args: argparse.Namespace, train: list[TicketRecord]):
    if args.method == "tfidf":
        return TfidfDuplicateDetector(combine=args.combine).fit(train)
    if not args.model_dir:
        raise SystemExit("--model-dir is required when --method sbert")
    return SbertDuplicateDetector.load(args.model_dir, combine=args.combine)


def collect_top1_errors(detector, queries: list[TicketRecord], candidates: list[TicketRecord], *, top_k: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    top_k = max(top_k, 1)
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidates)
        if not relevant:
            continue
        ranking = detector.rank(query, candidates, top_k=top_k)
        best = ranking[0] if ranking else None
        if best and best.ticket_id in relevant:
            continue
        rows.append(
            {
                "query_id": query.ticket_id,
                "query_title": query.title,
                "relevant_ids": " ".join(sorted(relevant)),
                "best_candidate_id": "" if best is None else best.ticket_id,
                "best_score": "" if best is None else f"{best.score:.6f}",
                "best_title_score": "" if best is None else f"{best.title_score:.6f}",
                "best_content_score": "" if best is None else f"{best.content_score:.6f}",
                "best_title": "" if best is None else best.title,
                "top_ranked_ids": " ".join(row.ticket_id for row in ranking),
            }
        )
    return rows


def write_errors_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_title",
        "relevant_ids",
        "best_candidate_id",
        "best_score",
        "best_title_score",
        "best_content_score",
        "best_title",
        "top_ranked_ids",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_evaluable_queries(test: list[TicketRecord], train: list[TicketRecord]) -> int:
    return sum(1 for query in test if relevant_duplicate_ids(query, train))


if __name__ == "__main__":
    raise SystemExit(main())
