"""
Perfetto Trace Post-Processor for MalthusJAX

Parses Perfetto JSON traces and extracts dispatch timing statistics
for JAX operations in MalthusJAX evolutionary algorithms.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """Represents a single trace event from Perfetto."""

    name: str
    category: str
    phase: str  # 'B' = begin, 'E' = end, 'X' = complete, 'i' = instant
    timestamp_us: float
    duration_us: float = 0.0
    thread_id: int = 0
    process_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class OperatorStats:
    """Statistics for a traced operator."""

    name: str
    call_count: int = 0
    total_duration_us: float = 0.0
    min_duration_us: float = float("inf")
    max_duration_us: float = 0.0
    durations: list[float] = field(default_factory=list)

    @property
    def avg_duration_us(self) -> float:
        return self.total_duration_us / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.avg_duration_us / 1000.0

    def add_duration(self, duration_us: float) -> None:
        self.call_count += 1
        self.total_duration_us += duration_us
        self.min_duration_us = min(self.min_duration_us, duration_us)
        self.max_duration_us = max(self.max_duration_us, duration_us)
        self.durations.append(duration_us)


@dataclass
class TraceAnalysis:
    """Complete analysis of a Perfetto trace."""

    trace_path: Path
    total_duration_us: float
    operator_stats: dict[str, OperatorStats]
    kernel_stats: dict[str, OperatorStats]
    dispatch_events: list[TraceEvent]
    metadata: dict = field(default_factory=dict)

    def get_dispatch_overhead_estimate(self) -> float:
        """
        Estimate dispatch overhead as percentage of total time.

        Returns the percentage of time spent in non-kernel operations.
        """
        total_kernel_time = sum(s.total_duration_us for s in self.kernel_stats.values())
        if self.total_duration_us > 0:
            return (1 - total_kernel_time / self.total_duration_us) * 100
        return 0.0

    def get_operator_breakdown(self) -> dict[str, float]:
        """Get percentage breakdown by operator."""
        if self.total_duration_us == 0:
            return {}
        return {
            name: (stats.total_duration_us / self.total_duration_us) * 100
            for name, stats in self.operator_stats.items()
        }

    def to_dict(self) -> dict:
        """Convert analysis to dictionary for JSON serialization."""
        return {
            "trace_path": str(self.trace_path),
            "total_duration_ms": self.total_duration_us / 1000,
            "dispatch_overhead_pct": self.get_dispatch_overhead_estimate(),
            "operators": {
                name: {
                    "call_count": stats.call_count,
                    "total_ms": stats.total_duration_us / 1000,
                    "avg_ms": stats.avg_duration_ms,
                    "min_ms": stats.min_duration_us / 1000,
                    "max_ms": stats.max_duration_us / 1000,
                }
                for name, stats in self.operator_stats.items()
            },
            "kernels": {
                name: {
                    "call_count": stats.call_count,
                    "total_ms": stats.total_duration_us / 1000,
                    "avg_ms": stats.avg_duration_ms,
                }
                for name, stats in self.kernel_stats.items()
            },
            "metadata": self.metadata,
        }


class PerfettoTraceParser:
    """Parser for Perfetto JSON trace files."""

    # Known JAX-related categories
    JAX_CATEGORIES = {"jax", "XLA", "JAX", "xla", "gpu", "cuda", "metal"}

    # Operator name patterns to identify MalthusJAX operators
    MALTHUSJAX_OPERATORS = {
        "traced_selection",
        "traced_crossover",
        "traced_mutation",
        "traced_fitness",
        "evolution_step",
        "TournamentSelection",
        "ArithmeticCrossover",
        "GaussianMutation",
    }

    def __init__(self, trace_path: Path | str):
        self.trace_path = Path(trace_path)
        self.events: list[TraceEvent] = []

    def parse(self) -> list[TraceEvent]:
        """Parse the trace file and return list of events."""
        trace_data = self._load_trace_file()

        if isinstance(trace_data, dict):
            # Chrome trace format with metadata
            events_data = trace_data.get("traceEvents", [])
        elif isinstance(trace_data, list):
            # Raw event array
            events_data = trace_data
        else:
            raise ValueError(f"Unexpected trace format: {type(trace_data)}")

        self.events = []
        for event in events_data:
            if not isinstance(event, dict):
                continue

            parsed = self._parse_event(event)
            if parsed:
                self.events.append(parsed)

        return self.events

    def _load_trace_file(self) -> Any:
        """Load trace file, handling both plain JSON and gzipped formats."""
        if self.trace_path.suffix == ".gz" or str(self.trace_path).endswith(".json.gz"):
            with gzip.open(self.trace_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(self.trace_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _parse_event(self, event: dict) -> TraceEvent | None:
        """Parse a single trace event."""
        phase = event.get("ph", "")
        if phase not in ("B", "E", "X", "i", "C"):
            return None

        name = event.get("name", "")
        category = event.get("cat", "")
        timestamp = event.get("ts", 0)
        duration = event.get("dur", 0)
        tid = event.get("tid", 0)
        pid = event.get("pid", 0)
        args = event.get("args", {})

        return TraceEvent(
            name=name,
            category=category,
            phase=phase,
            timestamp_us=timestamp,
            duration_us=duration,
            thread_id=tid,
            process_id=pid,
            args=args,
        )

    def analyze(self) -> TraceAnalysis:
        """Analyze parsed events and return statistics."""
        if not self.events:
            self.parse()

        operator_stats: dict[str, OperatorStats] = {}
        kernel_stats: dict[str, OperatorStats] = {}
        dispatch_events: list[TraceEvent] = []

        # Track begin/end pairs for duration calculation
        begin_stack: dict[tuple[int, str], TraceEvent] = {}

        min_ts = float("inf")
        max_ts = 0.0

        for event in self.events:
            # Track total duration
            ts_end = event.timestamp_us + event.duration_us
            min_ts = min(min_ts, event.timestamp_us)
            max_ts = max(max_ts, ts_end)

            # Handle complete events (phase 'X')
            if event.phase == "X" and event.duration_us > 0:
                self._categorize_event(
                    event, operator_stats, kernel_stats, dispatch_events
                )

            # Handle begin/end pairs
            elif event.phase == "B":
                key = (event.thread_id, event.name)
                begin_stack[key] = event

            elif event.phase == "E":
                key = (event.thread_id, event.name)
                if key in begin_stack:
                    begin_event = begin_stack.pop(key)
                    duration = event.timestamp_us - begin_event.timestamp_us
                    complete_event = TraceEvent(
                        name=event.name,
                        category=event.category,
                        phase="X",
                        timestamp_us=begin_event.timestamp_us,
                        duration_us=duration,
                        thread_id=event.thread_id,
                        process_id=event.process_id,
                        args={**begin_event.args, **event.args},
                    )
                    self._categorize_event(
                        complete_event, operator_stats, kernel_stats, dispatch_events
                    )

        total_duration = max_ts - min_ts if max_ts > min_ts else 0.0

        return TraceAnalysis(
            trace_path=self.trace_path,
            total_duration_us=total_duration,
            operator_stats=operator_stats,
            kernel_stats=kernel_stats,
            dispatch_events=dispatch_events,
            metadata={
                "event_count": len(self.events),
                "min_timestamp_us": min_ts,
                "max_timestamp_us": max_ts,
            },
        )

    def _categorize_event(
        self,
        event: TraceEvent,
        operator_stats: dict[str, OperatorStats],
        kernel_stats: dict[str, OperatorStats],
        dispatch_events: list[TraceEvent],
    ) -> None:
        """Categorize an event into operators, kernels, or dispatch."""
        name = event.name
        category = event.category

        # Check if it's a MalthusJAX operator
        is_operator = any(op in name for op in self.MALTHUSJAX_OPERATORS)

        # Check if it's a JAX/XLA kernel
        is_kernel = (
            "kernel" in name.lower()
            or "xla" in category.lower()
            or "gpu" in category.lower()
            or "cuda" in category.lower()
            or "metal" in category.lower()
        )

        # Check if it's a dispatch event
        is_dispatch = "dispatch" in name.lower() or "launch" in name.lower()

        if is_operator:
            if name not in operator_stats:
                operator_stats[name] = OperatorStats(name=name)
            operator_stats[name].add_duration(event.duration_us)

        if is_kernel:
            if name not in kernel_stats:
                kernel_stats[name] = OperatorStats(name=name)
            kernel_stats[name].add_duration(event.duration_us)

        if is_dispatch:
            dispatch_events.append(event)


def analyze_trace_file(trace_path: Path | str) -> TraceAnalysis:
    """Convenience function to analyze a trace file."""
    parser = PerfettoTraceParser(trace_path)
    return parser.analyze()


def analyze_trace_directory(trace_dir: Path | str) -> list[TraceAnalysis]:
    """Analyze all trace files in a directory."""
    trace_dir = Path(trace_dir)
    analyses = []

    for trace_file in trace_dir.glob("*.json*"):
        try:
            analysis = analyze_trace_file(trace_file)
            analyses.append(analysis)
        except Exception as e:
            print(f"Warning: Failed to parse {trace_file}: {e}")

    return analyses


def generate_trace_summary(analyses: list[TraceAnalysis], output_path: Path) -> None:
    """Generate a summary report from multiple trace analyses."""
    if not analyses:
        print("No analyses to summarize")
        return

    lines = [
        "=" * 80,
        "Perfetto Trace Summary",
        "=" * 80,
        "",
        f"Analyzed {len(analyses)} trace files",
        "",
    ]

    # Aggregate operator statistics
    all_operators: dict[str, list[float]] = defaultdict(list)
    all_kernels: dict[str, list[float]] = defaultdict(list)

    for analysis in analyses:
        for name, stats in analysis.operator_stats.items():
            all_operators[name].extend(stats.durations)
        for name, stats in analysis.kernel_stats.items():
            all_kernels[name].extend(stats.durations)

    lines.extend(
        [
            "-" * 80,
            "OPERATOR TIMING (aggregated across traces)",
            "-" * 80,
            f"{'Operator':<40} {'Count':<10} {'Avg (ms)':<12} {'Total (ms)':<12}",
            "-" * 80,
        ]
    )

    for name, durations in sorted(all_operators.items(), key=lambda x: -sum(x[1])):
        count = len(durations)
        total_ms = sum(durations) / 1000
        avg_ms = total_ms / count if count > 0 else 0
        lines.append(f"{name[:40]:<40} {count:<10} {avg_ms:<12.3f} {total_ms:<12.3f}")

    lines.extend(
        [
            "",
            "-" * 80,
            "KERNEL TIMING (aggregated across traces)",
            "-" * 80,
            f"{'Kernel':<40} {'Count':<10} {'Avg (ms)':<12} {'Total (ms)':<12}",
            "-" * 80,
        ]
    )

    for name, durations in sorted(
        all_kernels.items(), key=lambda x: -sum(x[1])
    )[:20]:  # Top 20 kernels
        count = len(durations)
        total_ms = sum(durations) / 1000
        avg_ms = total_ms / count if count > 0 else 0
        lines.append(f"{name[:40]:<40} {count:<10} {avg_ms:<12.3f} {total_ms:<12.3f}")

    lines.append("=" * 80)

    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"Summary saved to: {output_path}")
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python trace_processor.py <trace_file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        analysis = analyze_trace_file(path)
        print(json.dumps(analysis.to_dict(), indent=2))
    elif path.is_dir():
        analyses = analyze_trace_directory(path)
        generate_trace_summary(analyses, path / "trace_summary.txt")
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
