from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import statistics


@dataclass
class RunResult:
    seed: int
    status: str  # e.g., "success", "failure", "timeout", "error"
    metrics: Dict[str, float]
    history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    timings: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "metrics": self.metrics,
            "history": self.history,
            "artifacts": self.artifacts,
            "duration_seconds": self.duration_seconds,
            "timings": self.timings,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        d = dict(data)
        created = d.get("created_at")
        if isinstance(created, str):
            d["created_at"] = datetime.fromisoformat(created)
        elif created is None:
            d["created_at"] = datetime.now(timezone.utc)
        return cls(
            seed=d["seed"],
            status=d["status"],
            metrics=d.get("metrics", {}),
            history=d.get("history", []),
            artifacts=d.get("artifacts", {}),
            duration_seconds=d.get("duration_seconds"),
            timings=d.get("timings"),
            error=d.get("error"),
            created_at=d["created_at"],
        )

    @classmethod
    def from_json(cls, data: str) -> "RunResult":
        return cls.from_dict(json.loads(data))


@dataclass
class ExperimentResult:
    name: str
    runs: List[RunResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "0.1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "runs": [r.to_dict() for r in self.runs],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        d = dict(data)
        created = d.get("created_at")
        if isinstance(created, str):
            d["created_at"] = datetime.fromisoformat(created)
        elif created is None:
            d["created_at"] = datetime.now(timezone.utc)
        runs = [RunResult.from_dict(r) for r in d.get("runs", [])]
        return cls(
            name=d["name"],
            runs=runs,
            metadata=d.get("metadata", {}),
            created_at=d["created_at"],
            schema_version=d.get("schema_version", "0.1"),
        )

    def combined_history(self, seed_field: str = "seed") -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for run in self.runs:
            for row in run.history:
                combined = {**row, seed_field: run.seed}
                rows.append(combined)
        return rows

    @property
    def canonical_summary(self) -> Dict[str, Any]:
        if not self.runs:
            return {}
        return self.runs[0].metrics

    def aggregated_summary(self) -> Dict[str, Dict[str, float]]:
        # Collect numeric metrics across runs
        agg: Dict[str, List[float]] = {}
        for r in self.runs:
            for k, v in r.metrics.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                agg.setdefault(k, []).append(val)

        summary: Dict[str, Dict[str, float]] = {}
        for k, vals in agg.items():
            if not vals:
                continue
            mean = statistics.mean(vals)
            med = statistics.median(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[k] = {"mean": mean, "median": med, "stdev": stdev}
        return summary
