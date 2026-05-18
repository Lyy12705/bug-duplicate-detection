#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.rerank import RerankConfig, parse_field_weights, rerank_ranked_tickets
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector
from duplicate_ticket_detection.splits import train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import RankedTicket, TfidfDuplicateDetector


METADATA_FIELDS = ("product", "component", "severity", "priority")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Top-k duplicate recommendations for engineer review instead of automatic classification."
    )
    parser.add_argument("--tickets", default="data/mozilla_firefox_duplicates.csv")
    parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    parser.add_argument("--model-dir", help="Saved SBERT model directory. Required when --method sbert")
    parser.add_argument("--combine", default="weighted:0.6,0.4")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-ticket-id", help="Export recommendations for one existing ticket")
    parser.add_argument("--query-json", help="Export recommendations for one new ticket JSON")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply metadata reranking. Use --no-rerank to disable.",
    )
    parser.add_argument("--rerank-fields", nargs="*", default=["component:0.02"])
    parser.add_argument("--rerank-candidate-pool", type=int, default=50)
    parser.add_argument("--output-csv", default="reports/duplicate_review_queue.csv")
    parser.add_argument("--summary-md", default="reports/duplicate_review_queue_summary.md")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    queries, candidates, detector, mode = prepare_inputs(args, tickets)
    rerank_config = build_rerank_config(args)
    rows = build_review_rows(
        detector,
        queries,
        candidates,
        top_k=args.top_k,
        rerank_config=rerank_config,
    )
    write_review_csv(rows, args.output_csv)
    write_summary(rows, args.summary_md, mode=mode, top_k=args.top_k)

    print(f"mode={mode}")
    print(f"queries={len(queries)}")
    print(f"candidates={len(candidates)}")
    print(f"recommendation_rows={len(rows)}")
    print(f"output_csv={args.output_csv}")
    print(f"summary_md={args.summary_md}")
    return 0


def prepare_inputs(args: argparse.Namespace, tickets: list[TicketRecord]):
    if args.query_ticket_id or args.query_json:
        detector = build_detector(args, tickets)
        query = load_single_query(args, tickets)
        return [query], tickets, detector, "single_query"

    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    detector = build_detector(args, split.train)
    return split.test, split.train, detector, split.name


def build_detector(args: argparse.Namespace, train_tickets: list[TicketRecord]):
    if args.method == "tfidf":
        return TfidfDuplicateDetector(combine=args.combine).fit(train_tickets)
    if not args.model_dir:
        raise SystemExit("--model-dir is required when --method sbert")
    return SbertDuplicateDetector.load(args.model_dir, combine=args.combine)


def load_single_query(args: argparse.Namespace, tickets: list[TicketRecord]) -> TicketRecord:
    if args.query_ticket_id:
        for ticket in tickets:
            if ticket.ticket_id == args.query_ticket_id:
                return ticket
        raise SystemExit(f"query ticket not found: {args.query_ticket_id}")
    if args.query_json:
        return ticket_from_json(args.query_json)
    raise SystemExit("Provide --query-ticket-id or --query-json, or omit both to export a validation review queue.")


def ticket_from_json(path: str | Path) -> TicketRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--query-json must contain a JSON object")
    title = str(data.get("title") or data.get("summary") or data.get("subject") or "")
    content = str(data.get("description") or data.get("content") or data.get("body") or "")
    ticket_id = str(data.get("ticket_id") or data.get("bug_id") or data.get("id") or "new_ticket")
    fields = {str(key): "" if value is None else str(value) for key, value in data.items()}
    return TicketRecord(
        ticket_id=ticket_id,
        title=title,
        content=content,
        duplicate_of=None,
        duplicate_group=None,
        fields=fields,
    )


def build_rerank_config(args: argparse.Namespace) -> RerankConfig | None:
    if not args.rerank:
        return None
    return RerankConfig(
        field_weights=parse_field_weights(args.rerank_fields),
        candidate_pool=args.rerank_candidate_pool,
    )


def build_review_rows(
    detector,
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    *,
    top_k: int,
    rerank_config: RerankConfig | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    top_k = max(1, top_k)
    candidate_by_id = {ticket.ticket_id: ticket for ticket in candidates}
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidates)
        ranking = detector.rank(query, candidates, top_k=max(top_k, rerank_pool_size(rerank_config, top_k)))
        if rerank_config is not None:
            ranking = rerank_ranked_tickets(query, ranking, candidates, rerank_config, top_k=top_k)
        else:
            ranking = ranking[:top_k]
        rows.extend(review_rows_for_query(query, ranking, candidate_by_id, relevant))
    return rows


def rerank_pool_size(rerank_config: RerankConfig | None, top_k: int) -> int:
    if rerank_config is None:
        return top_k
    if rerank_config.candidate_pool <= 0:
        return max(top_k, 50)
    return max(top_k, rerank_config.candidate_pool)


def review_rows_for_query(
    query: TicketRecord,
    ranking: list[RankedTicket],
    candidate_by_id: dict[str, TicketRecord],
    relevant: set[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    first_score = ranking[0].score if ranking else 0.0
    second_score = ranking[1].score if len(ranking) > 1 else first_score
    top_margin = first_score - second_score
    relevant_in_top_k = bool(relevant and any(row.ticket_id in relevant for row in ranking))
    for rank, ranked in enumerate(ranking, start=1):
        candidate = candidate_by_id.get(ranked.ticket_id)
        matches = metadata_matches(query, candidate)
        rows.append(
            {
                "query_id": query.ticket_id,
                "query_title": query.title,
                "query_product": field_value(query, "product"),
                "query_component": field_value(query, "component"),
                "rank": str(rank),
                "candidate_id": ranked.ticket_id,
                "candidate_title": ranked.title,
                "candidate_score": f"{ranked.score:.6f}",
                "title_score": f"{ranked.title_score:.6f}",
                "content_score": f"{ranked.content_score:.6f}",
                "top1_margin": f"{top_margin:.6f}",
                "candidate_product": field_value(candidate, "product"),
                "candidate_component": field_value(candidate, "component"),
                "product_match": str(matches["product"]).lower(),
                "component_match": str(matches["component"]).lower(),
                "severity_match": str(matches["severity"]).lower(),
                "priority_match": str(matches["priority"]).lower(),
                "metadata_match_count": str(sum(matches.values())),
                "review_priority": review_priority(rank=rank, metadata_match_count=sum(matches.values()), top1_margin=top_margin),
                "known_relevant_ids": " ".join(sorted(relevant)),
                "candidate_is_known_duplicate": str(ranked.ticket_id in relevant).lower(),
                "known_duplicate_in_top_k": str(relevant_in_top_k).lower(),
            }
        )
    return rows


def metadata_matches(query: TicketRecord, candidate: TicketRecord | None) -> dict[str, bool]:
    return {field: same_field(query, candidate, field) for field in METADATA_FIELDS}


def review_priority(*, rank: int, metadata_match_count: int, top1_margin: float) -> str:
    if rank == 1 and (metadata_match_count >= 2 or top1_margin >= 0.03):
        return "high"
    if rank <= 3:
        return "medium"
    return "normal"


def write_review_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_title",
        "query_product",
        "query_component",
        "rank",
        "candidate_id",
        "candidate_title",
        "candidate_score",
        "title_score",
        "content_score",
        "top1_margin",
        "candidate_product",
        "candidate_component",
        "product_match",
        "component_match",
        "severity_match",
        "priority_match",
        "metadata_match_count",
        "review_priority",
        "known_relevant_ids",
        "candidate_is_known_duplicate",
        "known_duplicate_in_top_k",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], output_path: str | Path, *, mode: str, top_k: int) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_summary(rows, mode=mode, top_k=top_k) + "\n", encoding="utf-8")


def render_summary(rows: list[dict[str, str]], *, mode: str, top_k: int) -> str:
    query_ids = sorted({row["query_id"] for row in rows})
    rank1_rows = [row for row in rows if row["rank"] == "1"]
    evaluable = [row for row in rank1_rows if row["known_relevant_ids"]]
    top1_hits = sum(1 for row in evaluable if row["candidate_is_known_duplicate"] == "true")
    topk_hits = sum(1 for row in evaluable if row["known_duplicate_in_top_k"] == "true")
    priority_counts = Counter(row["review_priority"] for row in rows)
    lines = [
        "# Duplicate Review Queue Summary",
        "",
        f"- mode={mode}",
        f"- queries={len(query_ids)}",
        f"- top_k={top_k}",
        f"- recommendation_rows={len(rows)}",
        f"- high_priority_rows={priority_counts.get('high', 0)}",
        f"- medium_priority_rows={priority_counts.get('medium', 0)}",
    ]
    if evaluable:
        lines.extend(
            [
                f"- evaluable_duplicate_queries={len(evaluable)}",
                f"- top1_hit_rate={ratio(top1_hits, len(evaluable)):.4f}",
                f"- topk_hit_rate={ratio(topk_hits, len(evaluable)):.4f}",
            ]
        )
    lines.extend(["", "## Review Priority", "", "| Priority | Count |", "|---|---:|"])
    for priority, count in priority_counts.most_common():
        lines.append(f"| {priority} | {count} |")
    return "\n".join(lines)


def field_value(ticket: TicketRecord | None, field: str) -> str:
    if ticket is None or not ticket.fields:
        return ""
    return str(ticket.fields.get(field, ""))


def same_field(left: TicketRecord | None, right: TicketRecord | None, field: str) -> bool:
    left_value = field_value(left, field).strip().lower()
    right_value = field_value(right, field).strip().lower()
    return bool(left_value and left_value == right_value)


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
