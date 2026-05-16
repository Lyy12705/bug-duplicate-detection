"""Duplicate ticket detection toolkit."""

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets
from duplicate_ticket_detection.metrics import average_precision, mean_average_precision
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector
from duplicate_ticket_detection.triplets import TripletRecord, build_triplets

__all__ = [
    "TicketRecord",
    "TripletRecord",
    "TfidfDuplicateDetector",
    "average_precision",
    "build_triplets",
    "load_tickets",
    "mean_average_precision",
]
