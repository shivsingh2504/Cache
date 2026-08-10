from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from RL.trainer import EvaluationResult, Trainer

_TEMPLATE_PATH = Path(__file__).parent / "dashboard_template.html"
_PLACEHOLDER = "__DASHBOARD_DATA__"


def export_dashboard(
    trainer: Trainer,
    benchmark_results: dict[str, EvaluationResult],
    output_path: str | Path = "dashboard/results.html",
) -> Path:
    """Renders the results dashboard from real training/benchmark data.

    Reads `dashboard_template.html`, substitutes the embedded JSON data
    block with `trainer.eval_history` and `benchmark_results`, and writes
    a single self-contained HTML file to `output_path`.
    """
    data = {
        "meta": {
            "sample": False,
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "config": {
            "num_training_steps": trainer.config.num_training_steps,
            "eval_interval": trainer.config.eval_interval,
            "cache_capacity": trainer.environment.config.cache_capacity,
        },
        "trainingHistory": trainer.eval_history,
        "benchmark": {
            name: {"hit_rate": result.hit_rate, "num_episodes": result.num_episodes}
            for name, result in benchmark_results.items()
        },
    }

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = template.replace(_PLACEHOLDER, json.dumps(data, indent=2))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output