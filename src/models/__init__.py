from models.olmoe import OLMoELM
from models.llama import LlamaLM

__all__ = ["OLMoELM", "LlamaLM"]


# # Base models (like in the paper)
# LlamaLM("meta-llama/Llama-2-7b-hf")

# # Instruction-tuned (recommended - should follow QA format better)
# LlamaLM("meta-llama/Llama-2-7b-chat-hf")

# # Newer Llama 3 models
# LlamaLM("meta-llama/Meta-Llama-3-8B-Instruct")