from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    max_new_tokens: int = 40
    batch_size: int = 8
    device: str = "cpu"


@dataclass
class EvalConfig:
    probe_dir: str = "probes"
    output_dir: str = "output"
    models: list[ModelConfig] = field(default_factory=list)
    toxicity_model: str = "martin-ha/toxic-comment-model"
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    clip_model: str = "openai/clip-vit-base-patch32"
    toxicity_threshold: float = 0.5
    num_generations: int = 5
    seed: int = 42


def load_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = []
    for m in raw.get("models", []):
        models.append(ModelConfig(**m))

    return EvalConfig(
        probe_dir=raw.get("probe_dir", "probes"),
        output_dir=raw.get("output_dir", "output"),
        models=models,
        toxicity_model=raw.get("toxicity_model", "martin-ha/toxic-comment-model"),
        sentiment_model=raw.get("sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english"),
        clip_model=raw.get("clip_model", "openai/clip-vit-base-patch32"),
        toxicity_threshold=raw.get("toxicity_threshold", 0.5),
        num_generations=raw.get("num_generations", 5),
        seed=raw.get("seed", 42),
    )
