#!/usr/bin/env python3
"""CLI entry point for running fairlens evaluations."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fairlens.config import load_config
from fairlens.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run bias and toxicity evaluation on generative models")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.output:
        config.output_dir = args.output

    results = run_evaluation(config)

    for model_name, data in results.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        for dim, metrics in data["dimensions"].items():
            print(f"\n  [{dim}]")
            print(f"    Bias score:          {metrics['bias_score']:.4f}")
            tox = metrics["toxicity_parity"]
            print(f"    Toxicity disparity:  {tox['disparity']:.4f}")
            for g, r in tox["group_rates"].items():
                print(f"      {g:20s} {r:.4f}")
            sent = metrics["sentiment_parity"]
            print(f"    Sentiment disparity: {sent['disparity']:.4f}")
            for g, r in sent["group_rates"].items():
                print(f"      {g:20s} {r:.4f}")
            if any(v > 0 for v in metrics["stereotype_rates"].values()):
                print(f"    Stereotype rates:")
                for g, r in metrics["stereotype_rates"].items():
                    print(f"      {g:20s} {r:.4f}")


if __name__ == "__main__":
    main()
