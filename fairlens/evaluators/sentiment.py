from transformers import pipeline


class SentimentAnalyzer:
    """Sentiment classification for detecting valence bias across groups."""

    def __init__(
        self,
        model_id: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: int = -1,
    ):
        self._pipe = pipeline(
            "sentiment-analysis", model=model_id, device=device, truncation=True
        )

    def score(self, texts: list[str]) -> list[dict]:
        raw = self._pipe(texts, batch_size=16)
        results = []
        for r in raw:
            results.append({
                "label": r["label"],
                "score": r["score"],
                "positive": r["label"] == "POSITIVE",
            })
        return results

    def positive_rate(self, texts: list[str]) -> float:
        scores = self.score(texts)
        pos_count = sum(1 for s in scores if s["positive"])
        return pos_count / max(len(scores), 1)
