#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.metrics import average_precision
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector, SbertTrainingConfig
from duplicate_ticket_detection.splits import DatasetSplit, kfold_ticket_splits
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector, combine_similarity_scores
from duplicate_ticket_detection.triplets import build_triplets

DEFAULT_COMBINES = ("max", "title75", "content75", "mean")
DEFAULT_SBERT_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
)

@dataclass(frozen=True)
class Metrics:
    queries: int
    mean_average_precision: float
    top_1_accuracy: float
    top_k_hit_rate: float
    mean_reciprocal_rank: float

@dataclass(frozen=True)
class ResultRow:
    experiment: str
    method: str
    combine: str
    fold: str
    train_size: int
    test_size: int
    queries: int
    mean_average_precision: float
    top_1_accuracy: float
    top_k_hit_rate: float
    mean_reciprocal_rank: float
    base_model: str = ""
    epochs: int = 0
    max_triplets: int = 0
    seconds: float = 0.0

def main() -> int:
    parser = argparse.ArgumentParser(description="Run duplicate-ticket experiment comparison.")
    parser.add_argument("--tickets", default="data/mozilla_firefox_duplicates.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--combines", nargs="+", default=list(DEFAULT_COMBINES))
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--sbert-base-models", nargs="+", default=list(DEFAULT_SBERT_MODELS))
    parser.add_argument("--sbert-epochs", nargs="+", type=int, default=[2])
    parser.add_argument("--max-triplets", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-csv", default="reports/duplicate_experiment_results.csv")
    parser.add_argument("--output-md", default="reports/duplicate_experiment_results.md")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    splits = kfold_ticket_splits(tickets, n_splits=args.folds, seed=args.seed)
    print(f"tickets={len(tickets)} folds={len(splits)} top_k={args.top_k}", flush=True)
    print(f"combines={','.join(args.combines)}", flush=True)

    rows: list[ResultRow] = []
    if not args.skip_tfidf:
        rows.extend(run_tfidf_experiments(splits=splits, combines=args.combines, top_k=args.top_k))
    if not args.skip_sbert:
        rows.extend(run_sbert_experiments(
            splits=splits,
            combines=args.combines,
            top_k=args.top_k,
            base_models=args.sbert_base_models,
            epochs_values=args.sbert_epochs,
            max_triplets=args.max_triplets,
            batch_size=args.batch_size,
            margin=args.margin,
            warmup_steps=args.warmup_steps,
            local_files_only=args.local_files_only,
        ))

    write_results_csv(rows, args.output_csv)
    write_results_markdown(rows, args.output_md)
    print(f"csv={args.output_csv}", flush=True)
    print(f"markdown={args.output_md}", flush=True)
    return 0

def run_tfidf_experiments(*, splits: list[DatasetSplit], combines: list[str], top_k: int) -> list[ResultRow]:
    rows: list[ResultRow] = []
    fold_rows: list[ResultRow] = []
    for split in splits:
        print(f"[TF-IDF] {split.name}: fitting", flush=True)
        started = time.perf_counter()
        detector = TfidfDuplicateDetector(combine="max").fit(split.train)
        title_scores, content_scores = tfidf_score_matrices(detector, split.test, split.train)
        seconds = time.perf_counter() - started
        for combine in combines:
            scores = combine_similarity_scores(title_scores, content_scores, combine)
            metrics = score_matrix_metrics(split.test, split.train, scores, top_k=top_k)
            row = ResultRow(
                experiment=f"tfidf_{combine}", method="tfidf", combine=combine, fold=split.name,
                train_size=len(split.train), test_size=len(split.test), queries=metrics.queries,
                mean_average_precision=metrics.mean_average_precision,
                top_1_accuracy=metrics.top_1_accuracy,
                top_k_hit_rate=metrics.top_k_hit_rate,
                mean_reciprocal_rank=metrics.mean_reciprocal_rank,
                seconds=seconds,
            )
            rows.append(row)
            fold_rows.append(row)
            print_result(row)
    rows.extend(mean_rows(fold_rows))
    return rows

def run_sbert_experiments(
    *, splits: list[DatasetSplit], combines: list[str], top_k: int, base_models: list[str],
    epochs_values: list[int], max_triplets: int, batch_size: int, margin: float,
    warmup_steps: int, local_files_only: bool,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for base_model in base_models:
        for epochs in epochs_values:
            experiment_prefix = f"sbert_{model_slug(base_model)}_e{epochs}"
            fold_rows: list[ResultRow] = []
            for split in splits:
                print(f"[SBERT] {experiment_prefix} {split.name}: building triplets", flush=True)
                title_triplets = build_triplets(split.train, field="title", max_triplets=max_triplets)
                content_triplets = build_triplets(split.train, field="content", max_triplets=max_triplets)
                print(f"[SBERT] {experiment_prefix} {split.name}: title_triplets={len(title_triplets)} content_triplets={len(content_triplets)}", flush=True)
                config = SbertTrainingConfig(
                    base_model=base_model, epochs=epochs, batch_size=batch_size, margin=margin,
                    warmup_steps=warmup_steps, local_files_only=local_files_only,
                )
                detector = SbertDuplicateDetector(combine="max", config=config)
                started = time.perf_counter()
                detector.fit(title_triplets=title_triplets, content_triplets=content_triplets)
                title_scores, content_scores = sbert_score_matrices(detector, split.test, split.train)
                seconds = time.perf_counter() - started
                for combine in combines:
                    scores = combine_similarity_scores(title_scores, content_scores, combine)
                    metrics = score_matrix_metrics(split.test, split.train, scores, top_k=top_k)
                    row = ResultRow(
                        experiment=f"{experiment_prefix}_{combine}", method="sbert", combine=combine,
                        fold=split.name, train_size=len(split.train), test_size=len(split.test),
                        queries=metrics.queries, mean_average_precision=metrics.mean_average_precision,
                        top_1_accuracy=metrics.top_1_accuracy, top_k_hit_rate=metrics.top_k_hit_rate,
                        mean_reciprocal_rank=metrics.mean_reciprocal_rank, base_model=base_model,
                        epochs=epochs, max_triplets=max_triplets, seconds=seconds,
                    )
                    rows.append(row)
                    fold_rows.append(row)
                    print_result(row)
                del detector
                gc.collect()
                clear_torch_cache()
            rows.extend(mean_rows(fold_rows))
    return rows

def tfidf_score_matrices(detector: TfidfDuplicateDetector, queries: list[TicketRecord], candidates: list[TicketRecord]) -> tuple[np.ndarray, np.ndarray]:
    query_titles = detector.title_vectorizer.transform(safe_texts(ticket.title for ticket in queries))
    candidate_titles = detector.title_vectorizer.transform(safe_texts(ticket.title for ticket in candidates))
    query_contents = detector.content_vectorizer.transform(safe_texts(ticket.content for ticket in queries))
    candidate_contents = detector.content_vectorizer.transform(safe_texts(ticket.content for ticket in candidates))
    return (query_titles @ candidate_titles.T).toarray(), (query_contents @ candidate_contents.T).toarray()

def sbert_score_matrices(detector: SbertDuplicateDetector, queries: list[TicketRecord], candidates: list[TicketRecord]) -> tuple[np.ndarray, np.ndarray]:
    title_queries = detector.title_model.encode([ticket.title for ticket in queries], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    title_candidates = detector.title_model.encode([ticket.title for ticket in candidates], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    content_queries = detector.content_model.encode([ticket.content for ticket in queries], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    content_candidates = detector.content_model.encode([ticket.content for ticket in candidates], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(title_queries @ title_candidates.T), np.asarray(content_queries @ content_candidates.T)

def score_matrix_metrics(queries: list[TicketRecord], candidates: list[TicketRecord], scores: np.ndarray, *, top_k: int) -> Metrics:
    candidate_ids = [ticket.ticket_id for ticket in candidates]
    ap_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    top_1_hits = 0
    top_k_hits = 0
    for query_index, query in enumerate(queries):
        relevant = relevant_duplicate_ids(query, candidates)
        if not relevant:
            continue
        query_scores = scores[query_index].copy()
        for candidate_index, candidate in enumerate(candidates):
            if candidate.ticket_id == query.ticket_id:
                query_scores[candidate_index] = -np.inf
        ranked_ids = [candidate_ids[index] for index in np.argsort(-query_scores)]
        ap_scores.append(average_precision(ranked_ids, relevant))
        first_rank = first_relevant_rank(ranked_ids, relevant)
        reciprocal_ranks.append(0.0 if first_rank is None else 1 / first_rank)
        if ranked_ids and ranked_ids[0] in relevant:
            top_1_hits += 1
        if set(ranked_ids[:top_k]) & set(relevant):
            top_k_hits += 1
    queries_count = len(ap_scores)
    return Metrics(
        queries=queries_count,
        mean_average_precision=mean(ap_scores),
        top_1_accuracy=ratio(top_1_hits, queries_count),
        top_k_hit_rate=ratio(top_k_hits, queries_count),
        mean_reciprocal_rank=mean(reciprocal_ranks),
    )

def mean_rows(rows: list[ResultRow]) -> list[ResultRow]:
    grouped: dict[tuple[str, str, str, str, int, int], list[ResultRow]] = {}
    for row in rows:
        if row.fold == "mean":
            continue
        key = (row.experiment, row.method, row.combine, row.base_model, row.epochs, row.max_triplets)
        grouped.setdefault(key, []).append(row)
    means: list[ResultRow] = []
    for (experiment, method, combine, base_model, epochs, max_triplets), group_rows in grouped.items():
        evaluable = [row for row in group_rows if row.queries > 0]
        source_rows = evaluable or group_rows
        means.append(ResultRow(
            experiment=experiment, method=method, combine=combine, fold="mean",
            train_size=0, test_size=0,
            queries=round(sum(row.queries for row in source_rows) / len(source_rows)),
            mean_average_precision=mean(row.mean_average_precision for row in source_rows),
            top_1_accuracy=mean(row.top_1_accuracy for row in source_rows),
            top_k_hit_rate=mean(row.top_k_hit_rate for row in source_rows),
            mean_reciprocal_rank=mean(row.mean_reciprocal_rank for row in source_rows),
            base_model=base_model, epochs=epochs, max_triplets=max_triplets,
            seconds=sum(row.seconds for row in group_rows),
        ))
    return means

def write_results_csv(rows: list[ResultRow], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ResultRow.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})

def write_results_markdown(rows: list[ResultRow], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mean_results = sorted([row for row in rows if row.fold == "mean"], key=lambda row: row.mean_average_precision, reverse=True)
    lines = [
        "# Duplicate Ticket Experiment Results",
        "",
        "| Rank | Experiment | Method | Combine | Base model | Epochs | MAP | Top-1 | Top-k hit | MRR | Seconds |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(mean_results, start=1):
        lines.append(
            f"| {index} | {row.experiment} | {row.method} | {row.combine} | {row.base_model or '-'} | "
            f"{row.epochs or '-'} | {row.mean_average_precision:.4f} | {row.top_1_accuracy:.4f} | "
            f"{row.top_k_hit_rate:.4f} | {row.mean_reciprocal_rank:.4f} | {row.seconds:.1f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

def print_result(row: ResultRow) -> None:
    print(
        f"{row.fold},{row.experiment},queries={row.queries},MAP={row.mean_average_precision:.4f},"
        f"top1={row.top_1_accuracy:.4f},topk={row.top_k_hit_rate:.4f},MRR={row.mean_reciprocal_rank:.4f}",
        flush=True,
    )

def first_relevant_rank(ranked_ids: list[str], relevant_ids: Iterable[str]) -> int | None:
    relevant = set(relevant_ids)
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant:
            return rank
    return None

def safe_texts(texts: Iterable[str]) -> list[str]:
    values = [(text or "").strip() for text in texts]
    return [value if value else "__empty__" for value in values]

def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)

def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_")

def clear_torch_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    raise SystemExit(main())
