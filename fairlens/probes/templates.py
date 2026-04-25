from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class ProbeSet:
    dimension: str
    groups: list[str]
    templates: list[str]
    stereotypes: dict[str, list[str]] = field(default_factory=dict)

    def expand(self) -> list[dict]:
        """Expand templates across all groups, returning prompt dicts."""
        prompts = []
        for tpl in self.templates:
            for group in self.groups:
                prompts.append({
                    "dimension": self.dimension,
                    "group": group,
                    "template": tpl,
                    "prompt": tpl.replace("{group}", group),
                })
        return prompts


def load_probes(probe_dir: str | Path) -> list[ProbeSet]:
    probe_dir = Path(probe_dir)
    probe_sets = []

    for yaml_path in sorted(probe_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        ps = ProbeSet(
            dimension=raw["dimension"],
            groups=raw["groups"],
            templates=raw["templates"],
            stereotypes=raw.get("stereotypes", {}),
        )
        probe_sets.append(ps)

    return probe_sets
