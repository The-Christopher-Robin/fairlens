import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor


class CLIPBiasEvaluator:
    """
    Measures association bias between demographic terms and attribute terms
    using CLIP text embeddings. No images needed -- we compare text-text
    cosine similarity in CLIP's shared embedding space, following the
    approach from Wolfe & Caliskan (2022) adapted for quick audits.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self._device = device
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model = CLIPModel.from_pretrained(model_id).to(device)
        self._model.eval()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items() if k != "pixel_values"}
        with torch.no_grad():
            embeds = self._model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.cpu().numpy()

    def association_scores(
        self,
        group_terms: dict[str, list[str]],
        attribute_pairs: list[tuple[list[str], list[str]]],
    ) -> dict[str, dict[str, float]]:
        """
        For each group, compute differential association between two
        attribute clusters (e.g. pleasant vs unpleasant).

        Returns {group_name: {pair_label: score}} where positive score
        means stronger association with the first attribute set.
        """
        results = {}
        for group_name, terms in group_terms.items():
            group_embeds = self._encode_texts(terms)
            group_mean = group_embeds.mean(axis=0, keepdims=True)
            pair_scores = {}

            for idx, (attr_a, attr_b) in enumerate(attribute_pairs):
                embed_a = self._encode_texts(attr_a).mean(axis=0, keepdims=True)
                embed_b = self._encode_texts(attr_b).mean(axis=0, keepdims=True)

                sim_a = float(np.dot(group_mean, embed_a.T).squeeze())
                sim_b = float(np.dot(group_mean, embed_b.T).squeeze())
                pair_scores[f"pair_{idx}"] = sim_a - sim_b

            results[group_name] = pair_scores

        return results
