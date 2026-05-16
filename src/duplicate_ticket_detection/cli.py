from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids, write_tickets_csv
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector, SbertTrainingConfig
from duplicate_ticket_detection.splits import DatasetSplit, kfold_ticket_splits, train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import EvaluationRow, TfidfDuplicateDetector
from duplicate_ticket_detection.triplets import build_triplets, write_triplets_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duplicate ticket detection experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate duplicate ranking with MAP")
    _add_common_ticket_args(evaluate_parser)
    evaluate_parser.add_argument("--method", choices=("tfidf",), default="tfidf")
    evaluate_parser.add_argument("--combine", default="max")
    evaluate_parser.add_argument("--top-k", type=int)
    evaluate_parser.set_defaults(func=_evaluate)

    rank_parser = subparsers.add_parser("rank", help="Rank candidate duplicate tickets")
    _add_common_ticket_args(rank_parser)
    rank_parser.add_argument("--query-ticket-id")
    rank_parser.add_argument("--query-json", help="Path to a JSON object containing a new ticket")
    rank_parser.add_argument("--combine", default="max")
    rank_parser.add_argument("--threshold", type=float)
    rank_parser.add_argument("--top-k", type=int, default=10)
    rank_parser.set_defaults(func=_rank)

    train_parser = subparsers.add_parser("train-tfidf", help="Fit and save the TF-IDF baseline detector")
    _add_common_ticket_args(train_parser)
    train_parser.add_argument("--combine", default="max")
    train_parser.add_argument("--output", required=True)
    train_parser.set_defaults(func=_train_tfidf)

    split_parser = subparsers.add_parser("split-data", help="Write train/test CSV files for duplicate retrieval")
    _add_common_ticket_args(split_parser)
    split_parser.add_argument("--test-size", type=float, default=0.2)
    split_parser.add_argument("--seed", type=int, default=13)
    split_parser.add_argument("--train-output", required=True)
    split_parser.add_argument("--test-output", required=True)
    split_parser.set_defaults(func=_split_data)

    split_eval_parser = subparsers.add_parser("evaluate-split", help="Train on a split and evaluate held-out duplicate queries")
    _add_common_ticket_args(split_eval_parser)
    _add_experiment_args(split_eval_parser)
    split_eval_parser.add_argument("--test-size", type=float, default=0.2)
    split_eval_parser.add_argument("--seed", type=int, default=13)
    split_eval_parser.add_argument("--details", action="store_true")
    split_eval_parser.set_defaults(func=_evaluate_split)

    cv_parser = subparsers.add_parser("cross-validate", help="Run duplicate retrieval cross-validation")
    _add_common_ticket_args(cv_parser)
    _add_experiment_args(cv_parser)
    cv_parser.add_argument("--folds", type=int, default=5)
    cv_parser.add_argument("--seed", type=int, default=13)
    cv_parser.add_argument("--details", action="store_true")
    cv_parser.set_defaults(func=_cross_validate)

    sbert_train_parser = subparsers.add_parser("train-sbert", help="Fine-tune title/content SBERT models with triplets")
    _add_common_ticket_args(sbert_train_parser)
    _add_sbert_args(sbert_train_parser)
    sbert_train_parser.add_argument("--combine", default="max")
    sbert_train_parser.add_argument("--output-dir", required=True)
    sbert_train_parser.set_defaults(func=_train_sbert)

    sbert_rank_parser = subparsers.add_parser("rank-sbert", help="Rank candidates with saved SBERT models")
    _add_common_ticket_args(sbert_rank_parser)
    sbert_rank_parser.add_argument("--model-dir", required=True)
    sbert_rank_parser.add_argument("--query-ticket-id")
    sbert_rank_parser.add_argument("--query-json")
    sbert_rank_parser.add_argument("--combine", default="max")
    sbert_rank_parser.add_argument("--top-k", type=int, default=10)
    sbert_rank_parser.set_defaults(func=_rank_sbert)

    sbert_evaluate_parser = subparsers.add_parser("evaluate-sbert", help="Evaluate saved SBERT models with MAP and top-k hit rate")
    _add_common_ticket_args(sbert_evaluate_parser)
    sbert_evaluate_parser.add_argument("--model-dir", required=True)
    sbert_evaluate_parser.add_argument("--combine", default="max")
    sbert_evaluate_parser.add_argument("--top-k", type=int, default=3)
    sbert_evaluate_parser.set_defaults(func=_evaluate_sbert)

    triplet_parser = subparsers.add_parser("build-triplets", help="Build SBERT triplet fine-tuning data")
    _add_common_ticket_args(triplet_parser)
    triplet_parser.add_argument("--field", choices=("title", "content"), default="content")
    triplet_parser.add_argument("--max-triplets", type=int)
    triplet_parser.add_argument("--seed", type=int, default=13)
    triplet_parser.add_argument("--bidirectional", action="store_true")
    triplet_parser.add_argument("--output", required=True)
    triplet_parser.set_defaults(func=_build_triplets)

    args = parser.parse_args(argv)
    return args.func(args)


def _add_common_ticket_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tickets", required=True, help="CSV or JSONL ticket dataset")
    parser.add_argument(
        "--content-columns",
        nargs="*",
        help="Columns to concatenate as ticket content. Defaults to common bug-report fields.",
    )


def _add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    parser.add_argument("--combine", default="max")
    parser.add_argument("--top-k", type=int, default=10)
    _add_sbert_args(parser)


def _add_sbert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-triplets", type=int)
    parser.add_argument("--local-files-only", action="store_true", help="Load the base model from the local Hugging Face cache only")


def _evaluate(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector = TfidfDuplicateDetector(combine=args.combine)
    mean_ap, rows = detector.evaluate(tickets, top_k=args.top_k)

    _print_evaluation(tickets_count=len(tickets), mean_ap=mean_ap, rows=rows, top_k=args.top_k or 10)
    return 0


def _rank(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    query = _load_query(args, tickets)
    detector = TfidfDuplicateDetector(combine=args.combine).fit(tickets)
    ranking = detector.rank(query, tickets, top_k=args.top_k)

    if args.threshold is not None and ranking:
        print(f"is_duplicate={ranking[0].score >= args.threshold}")
        print(f"threshold={args.threshold:.4f}")

    print("rank,ticket_id,score,title_score,content_score,title")
    for index, row in enumerate(ranking, start=1):
        print(
            f"{index},{row.ticket_id},{row.score:.4f},{row.title_score:.4f},"
            f"{row.content_score:.4f},{_csv_cell(row.title)}"
        )
    return 0


def _train_tfidf(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector = TfidfDuplicateDetector(combine=args.combine).fit(tickets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output)
    print(f"saved={output}")
    return 0


def _split_data(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    write_tickets_csv(split.train, args.train_output)
    write_tickets_csv(split.test, args.test_output)
    print(f"tickets={len(tickets)}")
    print(f"train={len(split.train)}")
    print(f"test={len(split.test)}")
    print(f"evaluable_test_queries={_count_evaluable_queries(split.test, split.train)}")
    print(f"train_output={args.train_output}")
    print(f"test_output={args.test_output}")
    return 0


def _evaluate_split(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    mean_ap, rows = _run_split(split, args)

    print(f"split={split.name}")
    print(f"train={len(split.train)}")
    print(f"test={len(split.test)}")
    _print_evaluation(tickets_count=len(tickets), mean_ap=mean_ap, rows=rows, top_k=args.top_k, details=args.details)
    return 0


def _cross_validate(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    splits = kfold_ticket_splits(tickets, n_splits=args.folds, seed=args.seed)
    fold_summaries: list[dict[str, float]] = []

    print(f"tickets={len(tickets)}", flush=True)
    print(f"folds={len(splits)}", flush=True)
    print(f"method={args.method}", flush=True)
    print("fold,train,test,queries,MAP,top_1_accuracy,top_k_hit_rate,MRR", flush=True)
    for split in splits:
        print(f"{split.name}: training...", flush=True)
        mean_ap, rows = _run_split(split, args)
        print(f"{split.name}: evaluating done", flush=True)
        summary = _summarize_rows(mean_ap=mean_ap, rows=rows, top_k=args.top_k)
        fold_summaries.append(summary)
        print(
            f"{split.name},{len(split.train)},{len(split.test)},{len(rows)},"
            f"{summary['MAP']:.4f},{summary['top_1_accuracy']:.4f},"
            f"{summary['top_k_hit_rate']:.4f},{summary['MRR']:.4f}",
            flush=True,
        )
        if args.details:
            _print_query_rows(rows, top_k=args.top_k)

    if fold_summaries:
        evaluable_summaries = [summary for summary in fold_summaries if summary["queries"] > 0]
        averaged_summaries = evaluable_summaries or fold_summaries
        print("mean_over_evaluable_folds")
        for key in ("MAP", "top_1_accuracy", "top_k_hit_rate", "MRR"):
            print(f"{key}={sum(summary[key] for summary in averaged_summaries) / len(averaged_summaries):.4f}")
    return 0


def _train_sbert(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector, title_triplets, content_triplets = _train_sbert_detector(tickets, args)
    detector.save(args.output_dir)
    print(f"title_triplets={len(title_triplets)}")
    print(f"content_triplets={len(content_triplets)}")
    print(f"saved={args.output_dir}")
    return 0


def _rank_sbert(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    query = _load_query(args, tickets)
    detector = SbertDuplicateDetector.load(args.model_dir, combine=args.combine)
    ranking = detector.rank(query, tickets, top_k=args.top_k)

    print("rank,ticket_id,score,title_score,content_score,title")
    for index, row in enumerate(ranking, start=1):
        print(
            f"{index},{row.ticket_id},{row.score:.4f},{row.title_score:.4f},"
            f"{row.content_score:.4f},{_csv_cell(row.title)}"
        )
    return 0


def _evaluate_sbert(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector = SbertDuplicateDetector.load(args.model_dir, combine=args.combine)
    mean_ap, rows = detector.evaluate(tickets)

    _print_evaluation(tickets_count=len(tickets), mean_ap=mean_ap, rows=rows, top_k=args.top_k, details=True)
    return 0


def _build_triplets(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    triplets = build_triplets(
        tickets,
        field=args.field,
        max_triplets=args.max_triplets,
        seed=args.seed,
        bidirectional=args.bidirectional,
    )
    write_triplets_csv(triplets, args.output)
    print(f"triplets={len(triplets)}")
    print(f"output={args.output}")
    return 0


def _load_query(args: argparse.Namespace, tickets: list[TicketRecord]) -> TicketRecord:
    if bool(args.query_ticket_id) == bool(args.query_json):
        raise SystemExit("Provide exactly one of --query-ticket-id or --query-json")

    if args.query_ticket_id:
        for ticket in tickets:
            if ticket.ticket_id == args.query_ticket_id:
                return ticket
        raise SystemExit(f"Ticket id not found: {args.query_ticket_id}")

    with open(args.query_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("--query-json must contain a JSON object")
    return TicketRecord(
        ticket_id=str(data.get("ticket_id", "__query__")),
        title=str(data.get("title", data.get("summary", ""))),
        content=str(data.get("description", data.get("content", data.get("body", "")))),
        fields={str(key): str(value) for key, value in data.items()},
    )


def _run_split(split: DatasetSplit, args: argparse.Namespace) -> tuple[float, list[EvaluationRow]]:
    if args.method == "tfidf":
        detector = TfidfDuplicateDetector(combine=args.combine).fit(split.train)
        return detector.evaluate(split.test, candidates=split.train)

    detector, _, _ = _train_sbert_detector(split.train, args)
    return detector.evaluate(split.test, candidates=split.train)


def _train_sbert_detector(
    tickets: list[TicketRecord],
    args: argparse.Namespace,
) -> tuple[SbertDuplicateDetector, list, list]:
    title_triplets = build_triplets(tickets, field="title", max_triplets=args.max_triplets)
    content_triplets = build_triplets(tickets, field="content", max_triplets=args.max_triplets)
    config = SbertTrainingConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        warmup_steps=args.warmup_steps,
        local_files_only=args.local_files_only,
    )
    detector = SbertDuplicateDetector(combine=args.combine, config=config)
    detector.fit(title_triplets=title_triplets, content_triplets=content_triplets)
    return detector, title_triplets, content_triplets


def _count_evaluable_queries(test: list[TicketRecord], train: list[TicketRecord]) -> int:
    return sum(1 for query in test if relevant_duplicate_ids(query, train))


def _print_evaluation(
    *,
    tickets_count: int,
    mean_ap: float,
    rows: list[EvaluationRow],
    top_k: int,
    details: bool = True,
) -> None:
    summary = _summarize_rows(mean_ap=mean_ap, rows=rows, top_k=top_k)

    print(f"tickets={tickets_count}")
    print(f"queries_with_duplicates={len(rows)}")
    print(f"MAP={summary['MAP']:.4f}")
    print(f"top_1_accuracy={summary['top_1_accuracy']:.4f}")
    print(f"top_{max(top_k, 1)}_hit_rate={summary['top_k_hit_rate']:.4f}")
    print(f"MRR={summary['MRR']:.4f}")
    if details:
        _print_query_rows(rows, top_k=top_k)


def _print_query_rows(rows: list[EvaluationRow], *, top_k: int) -> None:
    top_k = max(top_k, 1)
    print("query_id,AP,first_duplicate_rank,relevant_ids,top_ranked_ids")
    for row in rows:
        top_ranked = " ".join(row.ranked_ids[:top_k])
        relevant = " ".join(row.relevant_ids)
        print(f"{row.query_id},{row.average_precision:.4f},{_first_duplicate_rank(row)},{relevant},{top_ranked}")


def _summarize_rows(*, mean_ap: float, rows: list[EvaluationRow], top_k: int) -> dict[str, float]:
    top_k = max(top_k, 1)
    top_1_hits = sum(1 for row in rows if row.ranked_ids and row.ranked_ids[0] in row.relevant_ids)
    top_k_hits = sum(1 for row in rows if set(row.ranked_ids[:top_k]) & set(row.relevant_ids))
    reciprocal_ranks = [_reciprocal_rank(row) for row in rows]
    return {
        "queries": float(len(rows)),
        "MAP": mean_ap,
        "top_1_accuracy": _ratio(top_1_hits, len(rows)),
        "top_k_hit_rate": _ratio(top_k_hits, len(rows)),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }


def _first_duplicate_rank(row: EvaluationRow) -> int | str:
    relevant = set(row.relevant_ids)
    for rank, ticket_id in enumerate(row.ranked_ids, start=1):
        if ticket_id in relevant:
            return rank
    return ""


def _reciprocal_rank(row: EvaluationRow) -> float:
    first_rank = _first_duplicate_rank(row)
    if not isinstance(first_rank, int):
        return 0.0
    return 1 / first_rank


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _csv_cell(value: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([value])
    return output.getvalue().strip()


if __name__ == "__main__":
    raise SystemExit(main())
