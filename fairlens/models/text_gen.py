import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseModel


class HFTextGenerator(BaseModel):
    """HuggingFace causal LM wrapper for text generation."""

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 8):
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device)
        self._model.eval()

    @property
    def name(self) -> str:
        return self._model_id.split("/")[-1]

    def generate(self, prompts: list[str], max_new_tokens: int = 40) -> list[str]:
        completions = []
        for i in range(0, len(prompts), self._batch_size):
            batch = prompts[i : i + self._batch_size]
            encoded = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)

            prompt_lengths = [
                encoded["attention_mask"][j].sum().item() for j in range(len(batch))
            ]

            with torch.no_grad():
                outputs = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            for j, output_ids in enumerate(outputs):
                new_tokens = output_ids[prompt_lengths[j] :]
                text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append(text.strip())

        return completions
