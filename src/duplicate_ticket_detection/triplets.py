from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from duplicate_ticket_detection.dataset import TicketRecord


@dataclass(frozen=True)
class TripletRecord:
    anchor_id: str
    positive_id: str
    negative_id: str
    anchor_text: str
    positive_text: str
    negative_text: str
    field: str


def build_triplets(
    tickets: Iterable[TicketRecord],
    *,
    field: str = "content",
    max_triplets: int | None = None,
    seed: int = 13,
    bidirectional: bool = False,
) -> list[TripletRecord]:
    """Build anchor/positive/negative examples from duplicate groups."""

    records = list(tickets)
    grouped: dict[str, list[TicketRecord]] = {}
    for ticket in records:
        if ticket.duplicate_group:
            grouped.setdefault(ticket.duplicate_group, []).append(ticket)

    grouped = {group: members for group, members in grouped.items() if len(members) >= 2}
    triplets: list[TripletRecord] = []

    for group, positives in grouped.items():
        negatives = [ticket for ticket in records if ticket.duplicate_group != group]
        if not negatives:
            continue

        for i, anchor in enumerate(positives):
            positive_range = range(len(positives)) if bidirectional else range(i + 1, len(positives))
            for j in positive_range:
                if i == j:
                    continue
                positive = positives[j]
                for negative in negatives:
                    triplets.append(
                        TripletRecord(
                            anchor_id=anchor.ticket_id,
                            positive_id=positive.ticket_id,
                            negative_id=negative.ticket_id,
                            anchor_text=anchor.text_for(field),
                            positive_text=positive.text_for(field),
                            negative_text=negative.text_for(field),
                            field=field,
                        )
                    )

    if max_triplets is not None and len(triplets) > max_triplets:
        rng = random.Random(seed)
        triplets = rng.sample(triplets, max_triplets)

    return triplets


def write_triplets_csv(triplets: Iterable[TripletRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "field",
        "anchor_id",
        "positive_id",
        "negative_id",
        "anchor_text",
        "positive_text",
        "negative_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for triplet in triplets:
            writer.writerow({field: getattr(triplet, field) for field in fields})
