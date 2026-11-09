"""Explicit versioned synthetic measurements for consumer regressions."""

from malthusjax.core.timing import execution_metadata
from malthusjax.core.timing_records import TimingRecord


def record(seconds=1.0, seed=0, order=0):
    metadata = execution_metadata()
    metadata["signature"] = {"synthetic": True}
    metadata["phases"] = {"initialization": 1.0, "compilation": 1.0, "host_postprocessing": 1.0}
    return TimingRecord.from_producer(
        {"execution": seconds},
        metadata,
        seed=seed,
        order=order,
        environment={"backend": "synthetic", "revision": "fixture"},
        duration=10.0,
    ).to_dict()
