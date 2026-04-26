import logging
from collections import defaultdict

from tqdm import tqdm

from .config import EvalConfig
from .probes.templates import load_probes
from .models.text_gen import HFTextGenerator
from .evaluators.toxicity import ToxicityScorer
from .evaluators.sentiment import SentimentAnalyzer
from .evaluators.stereotype import StereotypeDetector
from .metrics.fairness import (
    demographic_parity,
    equalized_odds_proxy,
    aggregate_bias_score,
)
from .reports.writer import write_json_report, write_csv_report

logger = logging.getLogger(__name__)


def run_evaluation(config: EvalConfig) -> dict:
    """Full evaluation pipeline: load models, generate, score, report."""
    probe_sets = load_probes(config.probe_dir)
    if not probe_sets:
        raise FileNotFoundError(f"No probe YAML files found in {config.probe_dir}")

    logger.info("Loaded %d probe dimensions", len(probe_sets))

    tox_scorer = ToxicityScorer(model_id=config.toxicity_model)
    sent_analyzer = SentimentAnalyzer(model_id=config.sentiment_model)
    stereo_detector = StereotypeDetector()

    all_results = {}

    for model_cfg in config.models:
        logger.info("Evaluating model: %s", model_cfg.hf_id)
        generator = HFTextGenerator(
            model_id=model_cfg.hf_id,
            device=model_cfg.device,
            batch_size=model_cfg.batch_size,
        )

        model_results = {"dimensions": {}}

        for ps in probe_sets:
            logger.info("  Dimension: %s (%d groups, %d templates)",
                        ps.dimension, len(ps.groups), len(ps.templates))

            stereo_detector.set_lexicon(ps.stereotypes)
            expanded = ps.expand()

            # group prompts by group name for per-group analysis
            group_prompts = defaultdict(list)
            for item in expanded:
                group_prompts[item["group"]].append(item["prompt"])

            group_completions: dict[str, list[str]] = {}
            group_tox_rates: dict[str, float] = {}
            group_pos_rates: dict[str, float] = {}
            group_stereo_rates: dict[str, float] = {}

            for group in tqdm(ps.groups, desc=f"  {ps.dimension}", leave=False):
                prompts = group_prompts[group]

                all_completions = []
                for prompt in prompts:
                    gens = generator.generate(
                        [prompt] * config.num_generations,
                        max_new_tokens=model_cfg.max_new_tokens,
                    )
                    all_completions.extend(gens)

                group_completions[group] = all_completions
                group_tox_rates[group] = tox_scorer.toxicity_rate(all_completions)
                group_pos_rates[group] = sent_analyzer.positive_rate(all_completions)
                group_stereo_rates[group] = stereo_detector.stereotype_rate(
                    all_completions, group
                )

            tox_parity = demographic_parity(group_tox_rates)
            sent_parity = demographic_parity(group_pos_rates)
            eq_odds = equalized_odds_proxy(group_pos_rates, group_tox_rates)

            dim_metrics = {
                "toxicity_parity": tox_parity,
                "sentiment_parity": sent_parity,
                "equalized_odds": eq_odds,
                "stereotype_rates": group_stereo_rates,
                "bias_score": aggregate_bias_score({
                    "tox": tox_parity,
                    "sent": sent_parity,
                    "eq": eq_odds,
                }),
                "sample_completions": {
                    g: comps[:3] for g, comps in group_completions.items()
                },
            }

            model_results["dimensions"][ps.dimension] = dim_metrics

        all_results[generator.name] = model_results

    json_path = write_json_report(all_results, config.output_dir)
    csv_path = write_csv_report(all_results, config.output_dir)
    logger.info("Reports written to %s", config.output_dir)

    return all_results
