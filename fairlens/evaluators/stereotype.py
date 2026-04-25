import re


class StereotypeDetector:
    """
    Checks whether generated text contains known stereotype-associated
    words for a given demographic group. Uses keyword matching against
    curated stereotype lexicons loaded from probe YAML files.
    """

    def __init__(self, stereotype_lexicon: dict[str, list[str]] | None = None):
        self._lexicon = stereotype_lexicon or {}

    def set_lexicon(self, lexicon: dict[str, list[str]]):
        self._lexicon = {k: [w.lower() for w in v] for k, v in lexicon.items()}

    def detect(self, text: str, group: str) -> dict:
        words = self._lexicon.get(group, [])
        if not words:
            return {"group": group, "matches": [], "stereotype_hit": False}

        text_lower = text.lower()
        matches = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
        return {
            "group": group,
            "matches": matches,
            "stereotype_hit": len(matches) > 0,
        }

    def stereotype_rate(self, texts: list[str], group: str) -> float:
        if not texts:
            return 0.0
        hits = sum(1 for t in texts if self.detect(t, group)["stereotype_hit"])
        return hits / len(texts)
