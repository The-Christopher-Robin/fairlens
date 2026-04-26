import json
from pathlib import Path
from fairlens.reports.writer import write_json_report, write_csv_report


def test_json_report(tmp_path):
    results = {
        "gpt2": {
            "dimensions": {
                "gender": {
                    "toxicity_parity": {"disparity": 0.1, "group_rates": {"man": 0.2, "woman": 0.3}},
                    "sentiment_parity": {"disparity": 0.15, "group_rates": {"man": 0.6, "woman": 0.45}},
                    "stereotype_rates": {"man": 0.1, "woman": 0.2},
                    "bias_score": 0.12,
                }
            }
        }
    }
    path = write_json_report(results, tmp_path, "test_report.json")
    assert path.exists()

    with open(path) as f:
        data = json.load(f)
    assert "generated_at" in data
    assert "gpt2" in data["models"]


def test_csv_report(tmp_path):
    results = {
        "gpt2": {
            "dimensions": {
                "gender": {
                    "toxicity_parity": {"disparity": 0.1, "group_rates": {"man": 0.2, "woman": 0.3}},
                    "sentiment_parity": {"disparity": 0.15, "group_rates": {"man": 0.6, "woman": 0.45}},
                    "stereotype_rates": {"man": 0.1, "woman": 0.2},
                    "bias_score": 0.12,
                }
            }
        }
    }
    path = write_csv_report(results, tmp_path, "test_report.csv")
    assert path is not None
    assert path.exists()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 group rows
    assert "model" in lines[0]
    assert "gpt2" in lines[1]


def test_json_report_creates_dir(tmp_path):
    nested = tmp_path / "a" / "b"
    path = write_json_report({"m": {}}, nested)
    assert path.exists()


def test_csv_report_empty(tmp_path):
    path = write_csv_report({}, tmp_path, "empty.csv")
    assert path is None
