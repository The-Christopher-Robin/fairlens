import json
import csv
from pathlib import Path
from datetime import datetime, timezone


def write_json_report(results: dict, output_dir: str | Path, filename: str = "report.json"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def write_csv_report(results: dict, output_dir: str | Path, filename: str = "report.csv"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model_data in results.items():
        for dimension, dim_metrics in model_data.get("dimensions", {}).items():
            tox_rates = dim_metrics.get("toxicity_parity", {}).get("group_rates", {})
            sent_rates = dim_metrics.get("sentiment_parity", {}).get("group_rates", {})
            stereo_rates = dim_metrics.get("stereotype_rates", {})

            for group in set(list(tox_rates.keys()) + list(sent_rates.keys())):
                rows.append({
                    "model": model_name,
                    "dimension": dimension,
                    "group": group,
                    "toxicity_rate": tox_rates.get(group, ""),
                    "positive_sentiment_rate": sent_rates.get(group, ""),
                    "stereotype_rate": stereo_rates.get(group, ""),
                    "toxicity_disparity": dim_metrics.get("toxicity_parity", {}).get("disparity", ""),
                    "sentiment_disparity": dim_metrics.get("sentiment_parity", {}).get("disparity", ""),
                    "bias_score": dim_metrics.get("bias_score", ""),
                })

    if not rows:
        return None

    path = output_dir / filename
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
