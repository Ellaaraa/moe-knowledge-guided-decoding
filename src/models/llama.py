import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaLM:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str | None = None,
        dtype=torch.bfloat16,
        max_length: int = 2048,
    ):
        """
        Llama model wrapper with same interface as OLMoELM.
        
        Args:
            model_name: HuggingFace model name. Options include:
                - "meta-llama/Llama-2-7b-hf" (base, used in paper)
                - "meta-llama/Llama-2-7b-chat-hf" (instruction-tuned)
                - "meta-llama/Meta-Llama-3-8B" (newer)
                - "meta-llama/Meta-Llama-3-8B-Instruct" (instruction-tuned)
            device: Device to use (defaults to cuda if available)
            dtype: Model dtype (bfloat16 recommended)
            max_length: Maximum sequence length
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not set (Llama doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

    @torch.no_grad()
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
