from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from models.olmoe import OLMoELM
from data.datasets import QAExample
from decoding.vanilla import build_vanilla_prompt


@dataclass
class KGDPrediction:
    """Prediction container for Knowledge-Guided Decoding."""
    id: str
    question: str
    prediction: str
    gold_answers: list[str]
    weight: float      # reward weight parameter (w in paper)
    num_chunks: int    # how many chunks were used
    reward_type: str   # 'similarity', 'entailment', or 'combined'


# ======================== Document Chunking ======================== #

def build_kgd_prompt(
    example: QAExample, 
    knowledge_chunks: List[str],
    tokenizer=None,
    use_chat_template: bool = True,
) -> str:
    """
    Short-form QA prompt with retrieved contexts.
    
    If tokenizer is provided and use_chat_template=True, applies the model's
    chat template (required for instruction-tuned models like Qwen2.5-Instruct).
    """
    # Use at most three contexts, in ranked order.
    contexts = [c.strip() for c in knowledge_chunks[:3] if c.strip()]
    context_block = "\n".join(contexts)

    user_message = (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_block}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {example.question}\n"
        "Answer:"
    )
    
    # Apply chat template for instruction-tuned models
    if tokenizer is not None and use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a factual question-answering assistant. Answer briefly with just the answer phrase."},
            {"role": "user", "content": user_message},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback to raw prompt if chat template fails
            pass
    
    return user_message


def build_bio_prompt(entity: str, knowledge_chunks: List[str]) -> str:
    """
    Long-form biography prompt template from the paper:

    [Context 1]
    {retrieved context 1}
    [Context 2]
    {retrieved context 2}
    [Context 3]
    {retrieved context 3}
    ---------------------
    Question: Tell me an at least 50 words bio of {entity}. You must summarize but
    not directly copy the answer words by words from the contexts.
    Answer:
    """
    contexts = [c.strip() for c in knowledge_chunks[:3] if c.strip()]
    # Pad with empty strings so indices 0,1,2 always exist
    while len(contexts) < 3:
        contexts.append("")

    return (
        "[Context 1]\n"
        f"{contexts[0]}\n"
        "[Context 2]\n"
        f"{contexts[1]}\n"
        "[Context 3]\n"
        f"{contexts[2]}\n"
        "---------------------\n"
        f"Question: Tell me an at least 50 words bio of {entity}. "
        "You must summarize but not directly copy the answer words by words from the contexts.\n"
        "Answer:"
    )


def chunk_documents(
    documents: List[str], 
    chunk_size: int = 500, 
    overlap: int = 100
) -> List[str]:
    """
    Chunk documents into overlapping segments.
    
    Args:
        documents: List of document strings
        chunk_size: Size of each chunk in characters (default: 500)
        overlap: Overlap between chunks in characters (default: 100)
    
    Returns:
        List of document chunks
    """
    chunks = []
    
    for doc in documents:
        if len(doc) <= chunk_size:
            chunks.append(doc)
            continue
        
        start = 0
        while start < len(doc):
            end = min(start + chunk_size, len(doc))
            chunk = doc[start:end]
            chunks.append(chunk)
            
            if end == len(doc):
                break
            
            # Move forward by (chunk_size - overlap)
            start += chunk_size - overlap
    
    return chunks


# ======================== Document Retrieval ======================== #

class DocumentRetriever:
    """
    Retriever using sentence-transformers for embedding-based retrieval.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
    
    def retrieve_chunks(
        self, 
        query: str, 
        chunks: List[str], 
        top_k: int = 3
    ) -> List[str]:
        """
        Retrieve top-k most similar chunks to the query.
        
        Args:
            query: Query string
            chunks: List of document chunks
            top_k: Number of chunks to retrieve
        
        Returns:
            List of top-k most similar chunks
        """
        if not chunks:
            return []
        
        # Encode query and chunks
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0), 
            chunk_embeddings, 
            dim=1
        )
        
        # Get top-k indices
        top_k = min(top_k, len(chunks))
        top_indices = torch.topk(similarities, k=top_k).indices
        retrieved = [chunks[idx] for idx in top_indices.cpu().numpy()]

        # Debug print to surface retrieval stats for each query.
        print(
            f"[KGD] Retrieved {len(retrieved)} chunk(s) (top_k={top_k}) "
            f"from {len(chunks)} candidate chunk(s) for query: '{query}'"
        )
        
        # Return top-k chunks
        return retrieved


# ======================== Reward Functions ======================== #

class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute_reward(
        self, 
        generated_text: str, 
        knowledge_chunks: List[str]
    ) -> float:
        """
        Compute reward for generated text given knowledge chunks.
        
        Args:
            generated_text: Current generated text (x_{<t} concatenated with x_t)
            knowledge_chunks: List of retrieved knowledge chunks
        
        Returns:
            Reward score
        """
        pass


class SimilarityReward(RewardFunction):
    """
    Semantic similarity-based reward using sentence embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize similarity reward with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
    
    def compute_reward(
        self, 
        generated_text: str, 
        knowledge_chunks: List[str]
    ) -> float:
        """
        Compute semantic similarity reward.
        
        Returns the maximum cosine similarity between the generated text
        and any knowledge chunk.
        """
        if not knowledge_chunks or not generated_text.strip():
            return 0.0
        
        # Encode generated text and knowledge chunks
        gen_embedding = self.model.encode(generated_text, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(knowledge_chunks, convert_to_tensor=True)
        
        # Compute cosine similarity with all chunks
        similarities = F.cosine_similarity(
            gen_embedding.unsqueeze(0), 
            chunk_embeddings, 
            dim=1
        )
        
        # Return maximum similarity
        return similarities.max().item()


class EntailmentReward(RewardFunction):
    """
    Entailment-based reward using NLI model.
    """
    
    def __init__(
        self, 
        model_name: str = "microsoft/deberta-base-mnli",
        alpha: float = 5.0,
        beta: float = 10.0
    ):
        """
        Initialize entailment reward with an NLI model.
        
        Args:
            model_name: Name of the NLI model
            alpha: Weight for contradiction penalty
            beta: Weight for entailment reward
        """
        self.nli_pipeline = pipeline(
            "text-classification", 
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        self.alpha = alpha
        self.beta = beta
    
    def compute_reward(
        self, 
        generated_text: str, 
        knowledge_chunks: List[str]
    ) -> float:
        """
        Compute entailment-based reward.
        
        Formula: r_entailment = β * P(entailment) - α * P(contradiction)
        """
        if not knowledge_chunks or not generated_text.strip():
            return 0.0
        
        rewards = []
        
        for chunk in knowledge_chunks:
            # NLI: premise=knowledge, hypothesis=generated_text
            try:
                result = self.nli_pipeline(
                    {"text": chunk, "text_pair": generated_text},
                    truncation=True
                )[0]
                
                # Get probabilities for all labels
                label = result['label']
                score = result['score']
                
                # Compute reward based on entailment/contradiction
                if 'ENTAILMENT' in label.upper():
                    reward = self.beta * score
                elif 'CONTRADICTION' in label.upper():
                    reward = -self.alpha * score
                else:  # NEUTRAL
                    reward = 0.0
                
                rewards.append(reward)
            except Exception as e:
                # If NLI fails, return 0 reward
                rewards.append(0.0)
        
        # Return maximum reward across chunks
        return max(rewards) if rewards else 0.0


class CombinedReward(RewardFunction):
    """
    Combined reward using both similarity and entailment.
    """
    
    def __init__(
        self,
        similarity_weight: float = 1.5,
        alpha: float = 5.0,
        beta: float = 10.0
    ):
        """
        Initialize combined reward.
        
        Args:
            similarity_weight: Weight for similarity reward (γ in paper)
            alpha: Weight for contradiction penalty in entailment
            beta: Weight for entailment reward
        """
        self.similarity_reward = SimilarityReward()
        self.entailment_reward = EntailmentReward(alpha=alpha, beta=beta)
        self.similarity_weight = similarity_weight
    
    def compute_reward(
        self, 
        generated_text: str, 
        knowledge_chunks: List[str]
    ) -> float:
        """
        Compute combined reward.
        
        Formula: r_combined = γ * r_similarity + r_entailment
        """
        if not knowledge_chunks or not generated_text.strip():
            return 0.0
        
        sim_reward = self.similarity_reward.compute_reward(
            generated_text, knowledge_chunks
        )
        ent_reward = self.entailment_reward.compute_reward(
            generated_text, knowledge_chunks
        )
        
        return self.similarity_weight * sim_reward + ent_reward


# ======================== KGD Algorithm ======================== #

def kgd_decode_single(
    model: OLMoELM,
    example: QAExample,
    reward_function: Optional[RewardFunction] = None,
    weight: float = 2.0,
    max_new_tokens: int = 32,
    top_m: int = 4,
    temperature: float = 0.0,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    top_k_chunks: int = 3,
    eos_penalty: float = 10.0,
    min_new_tokens: int = 5,
) -> KGDPrediction:
    """
    Decode one QAExample with Knowledge-Guided Decoding (Algorithm 1).
    
    Args:
        model: Language model
        example: QA example with question and context
        reward_function: Reward function to use (default: SimilarityReward)
        weight: Weight parameter w for reward (default: 2.0)
        max_new_tokens: Maximum number of tokens to generate (T in paper)
        top_m: Number of top tokens to consider for reward (default: 4)
        temperature: Sampling temperature (0.0 = greedy after reward)
        chunk_size: Size of document chunks in characters
        chunk_overlap: Overlap between chunks in characters
        top_k_chunks: Number of chunks to retrieve (N in paper, default: 3)
    
    Returns:
        KGDPrediction object with generated text and metadata
    """
    device = model.device
    tokenizer = model.tokenizer
    
    # Step 1: Retrieve relevant knowledge k = {k_i}_{i=1}^N from K based on query q
    # Use example context as documents
    docs = [example.context] if example.context else []
    
    if not docs:
        # No context, fall back to vanilla decoding
        base_prompt = build_vanilla_prompt(example)
        inputs = tokenizer(
            base_prompt,
            return_tensors="pt",
            max_length=model.max_length,
            truncation=True,
        ).to(device)
        
        output = model.generate(
            base_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        if "Answer:" in output:
            answer = output.split("Answer:", 1)[1].strip().split("\n")[0]
        else:
            answer = output.strip().split("\n")[-1]
        
        return KGDPrediction(
            id=example.id,
            question=example.question,
            prediction=answer,
            gold_answers=example.answers,
            weight=weight,
            num_chunks=0,
            reward_type="none",
        )
    
    # Chunk documents
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Retrieve top-k chunks
    retriever = DocumentRetriever()
    knowledge_chunks = retriever.retrieve_chunks(
        example.question, chunks, top_k=top_k_chunks
    )
    
    # Initialize reward function
    if reward_function is None:
        reward_function = SimilarityReward()
    
    # Determine reward type for logging
    reward_type = reward_function.__class__.__name__.replace("Reward", "").lower()
    
    # Step 2: Initialize generated tokens x_{<1} with q prepended with contexts k
    # For KGD we explicitly include retrieved knowledge chunks in the prompt.
    # Pass tokenizer to apply chat template for instruction-tuned models.
    base_prompt = build_kgd_prompt(example, knowledge_chunks, tokenizer=tokenizer)
    
    # Debug: print the prompt being passed to the model
    print(f"\n[KGD] === PROMPT ===\n{base_prompt}\n[KGD] === END PROMPT ===\n")
    
    # Tokenize base prompt
    inputs = tokenizer(
        base_prompt,
        return_tensors="pt",
        max_length=model.max_length,
        truncation=True,
    ).to(device)
    
    input_ids = inputs["input_ids"]  # (1, L)
    
    # Step 3-14: Generation loop (for t = 1 to T do)
    last_logits: Optional[torch.Tensor] = None
    for t in range(max_new_tokens):
        # Step 4: Compute language model logits z(x_t|x_{<t}) over vocabulary
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )
            base_logits = out.logits[:, -1, :]   # (1, vocab_size)
        
        # Step 5: Select top-m tokens V_m based on logit values
        top_m_actual = min(top_m, base_logits.size(-1))
        top_m_values, top_m_indices = torch.topk(base_logits[0], k=top_m_actual)
        
        # Step 6-9: For each x_t in V_m do
        rewards = []
        for idx in top_m_indices:
            # Step 7: y_t = [x_{<t}, x_t]
            candidate_token_id = idx.unsqueeze(0).unsqueeze(0)  # (1, 1)
            candidate_ids = torch.cat([input_ids, candidate_token_id], dim=1)
            
            # Decode to get text
            y_t = tokenizer.decode(candidate_ids[0], skip_special_tokens=True)
            
            # Step 8: Compute knowledge-guided reward r(y_t, k)
            reward = reward_function.compute_reward(y_t, knowledge_chunks)
            rewards.append(reward)
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, device=device, dtype=base_logits.dtype)
        
        # Step 9: Update logits: z_KGD(x_t|x_{<t}) = z(x_t|x_{<t}) + w * r(y_t, k)
        updated_logits = base_logits.clone()
        updated_logits[0, top_m_indices] = (
            base_logits[0, top_m_indices] + weight * rewards_tensor
        )
        
        # Penalize EOS token for the first min_new_tokens steps to prevent empty outputs
        if t < min_new_tokens and tokenizer.eos_token_id is not None:
            updated_logits[0, tokenizer.eos_token_id] -= eos_penalty
        
        last_logits = updated_logits[0].detach().cpu()
        
        # Step 11: Compute p_KGD(x_t|x_{<t}) by applying softmax over updated logits
        probs = torch.softmax(updated_logits, dim=-1)
        
        # Step 12: Sample next token x_t ~ p_KGD(x_t|x_{<t})
        if temperature > 0.0:
            probs = torch.softmax(updated_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)  # (1,) -> (1, 1)
        else:
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)  # (1, vocab) -> (1, 1)
        
        token_id = next_token_id.item()
        
        # Step 13: Update generated tokens x_{<t+1} = [x_{<t}, x_t]
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        # Check for EOS token (only after min_new_tokens)
        if token_id == tokenizer.eos_token_id and t >= min_new_tokens:
            break
    
    # Step 15: return Generated text x_{<T+1}
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Surface the final logits for debugging (print top tokens to keep output concise)
    if last_logits is not None:
        final_top_values, final_top_indices = torch.topk(last_logits, k=10)
        final_tokens = tokenizer.convert_ids_to_tokens(final_top_indices.tolist())
        final_info = ", ".join(
            f"{tok}:{val:.3f}" for tok, val in zip(final_tokens, final_top_values.tolist())
        )
        print(f"[KGD] Final logits top-10 tokens: {final_info}")
    
    # Extract answer
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", 1)[1].strip().split("\n")[0]
    else:
        answer = full_text.strip().split("\n")[-1]
    
    return KGDPrediction(
        id=example.id,
        question=example.question,
        prediction=answer,
        gold_answers=example.answers,
        weight=weight,
        num_chunks=len(knowledge_chunks),
        reward_type=reward_type,
    )


def kgd_decode(
    model: OLMoELM,
    examples: Iterable[QAExample],
    reward_type: str = "similarity",
    weight: float = 2.0,
    alpha: float = 5.0,
    beta: float = 10.0,
    max_new_tokens: int = 32,
    top_m: int = 4,
    temperature: float = 0.0,
    show_progress: bool = True,
) -> List[KGDPrediction]:
    """
    Run KGD on a collection of QAExample objects.
    
    Args:
        model: Language model
        examples: Iterable of QA examples
        reward_type: Type of reward ('similarity', 'entailment', 'combined')
        weight: Weight parameter w (for similarity) or γ (for combined)
        alpha: Contradiction penalty weight (for entailment/combined)
        beta: Entailment reward weight (for entailment/combined)
        max_new_tokens: Maximum tokens to generate
        top_m: Number of top tokens for reward computation
        temperature: Sampling temperature
        show_progress: Whether to show progress bar
    
    Returns:
        List of KGDPrediction objects
    """
    # Initialize reward function based on type
    if reward_type == "similarity":
        reward_function = SimilarityReward()
    elif reward_type == "entailment":
        reward_function = EntailmentReward(alpha=alpha, beta=beta)
    elif reward_type == "combined":
        reward_function = CombinedReward(
            similarity_weight=weight, alpha=alpha, beta=beta
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    iterator = examples
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            desc = f"Decoding (KGD-{reward_type}, w={weight})"
            iterator = tqdm(examples, desc=desc)
        except ImportError:
            iterator = examples
    
    preds: List[KGDPrediction] = []
    for ex in iterator:
        pred = kgd_decode_single(
            model,
            ex,
            reward_function=reward_function,
            weight=weight,
            max_new_tokens=max_new_tokens,
            top_m=top_m,
            temperature=temperature,
        )
        preds.append(pred)
    
    return preds
