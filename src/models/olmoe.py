import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class OLMoELM:
    def __init__(
        self,
        model_name="allenai/OLMoE-1B-7B-0125",
        device=None,
        dtype=torch.bfloat16,
        max_length=2048,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = OlmoeForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

    @torch.no_grad() # Disabled gradient calculations for faster inference

    def generate(self, prompt, max_new_tokens=64, temperature=0.0):
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