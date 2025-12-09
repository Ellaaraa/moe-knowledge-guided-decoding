import torch
from transformers import AutoTokenizer, OlmoeForCausalLM, BitsAndBytesConfig


class OLMoELM:
    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0125",
        device: str | None = None,
        dtype = torch.bfloat16,
        max_length: int = 2048,
        use_4bit: bool = False,  # Enable 4-bit quantization to save memory
    ):
        # Use GPU if available, else CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with memory optimization
        if use_4bit and self.device.startswith("cuda"):
            # Solution 2: 4-bit quantization (most memory-efficient)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            self.model = OlmoeForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Solution 1: Load directly to GPU
                low_cpu_mem_usage=True,  # Solution 1: Reduce RAM usage
            )
        elif self.device.startswith("cuda"):
            # Solution 1: Direct GPU loading without quantization
            self.model = OlmoeForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",  # Load directly to GPU, skip RAM
                low_cpu_mem_usage=True,  # Reduce RAM usage during loading
            )
        else:
            # For CPU, use standard loading
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
