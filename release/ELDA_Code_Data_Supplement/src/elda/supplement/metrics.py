"""Small-sample metrics derived from the realized Source--Demand object."""

from __future__ import annotations

import hashlib
import json
from collections import Counter


def sample_metrics(payload: dict) -> dict:
    fanouts = [int(row.get("fanout", 0)) for row in payload["source_nets"]]
    kinds = Counter(row["source_kind"] for row in payload["sources"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "cell_count": len(payload["cells"]),
        "demand_count": len(payload["demands"]),
        "source_count": len(payload["sources"]),
        "assignment_count": sum(fanouts),
        "source_fanout_histogram": {str(key): value for key, value in sorted(Counter(fanouts).items())},
        "source_kind_histogram": dict(sorted(kinds.items())),
        "mean_source_fanout": sum(fanouts) / max(1, len(fanouts)),
        "object_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
