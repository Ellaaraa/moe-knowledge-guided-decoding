import torch
from transformers import AutoTokenizer, OlmoeForCausalLM


class OLMoELM:
    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0125",
        device: str | None = None,
        dtype = torch.bfloat16,
        max_length: int = 2048,
    ):
        # Use GPU if available, else CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model (single device; good for cluster)
        self.model = OlmoeForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

    @torch.no_grad()  # Disable gradients for faster inference

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
