from transformers import pipeline


class ToxicityScorer:
    """Score text for toxicity using a HuggingFace classifier."""

    def __init__(self, model_id: str = "martin-ha/toxic-comment-model", device: int = -1):
        self._pipe = pipeline(
            "text-classification", model=model_id, device=device, truncation=True
        )

    def score(self, texts: list[str]) -> list[dict]:
        """
        Returns list of {label, score, is_toxic} for each text.
        The model outputs 'toxic' / 'non-toxic' labels.
        """
        raw = self._pipe(texts, batch_size=16)
        results = []
        for r in raw:
            label = r["label"].lower()
            is_toxic = label == "toxic"
            results.append({
                "label": r["label"],
                "score": r["score"],
                "is_toxic": is_toxic if r["score"] > 0.5 else not is_toxic,
            })
        return results

    def toxicity_rate(self, texts: list[str]) -> float:
        scores = self.score(texts)
        toxic_count = sum(1 for s in scores if s["is_toxic"])
        return toxic_count / max(len(scores), 1)
