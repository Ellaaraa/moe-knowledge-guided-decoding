# Knowledge-Guided Decoding (KGD) Implementation

This document describes the complete implementation of the KGD algorithm from the paper "Enhancing Factuality in Language Models through Knowledge-Guided Decoding".

## Overview

The implementation follows Algorithm 1 from the paper exactly, with all components specified in Section 5.3 (Experimental Details).

## Implementation Components

### 1. Document Chunking (`chunk_documents`)
- Chunks documents into 500 characters with 100 character overlap
- Handles documents shorter than chunk size gracefully
- Used to process long documents into manageable segments

### 2. Document Retrieval (`DocumentRetriever`)
- Uses `sentence-transformers/all-mpnet-base-v2` model for embeddings
- Retrieves top N=3 most similar chunks based on cosine similarity
- Efficient vector-based retrieval

### 3. Reward Functions

#### Similarity Reward (`SimilarityReward`)
- Computes semantic similarity using sentence embeddings
- Returns maximum cosine similarity between generated text and knowledge chunks
- Default weight: w=2.0

#### Entailment Reward (`EntailmentReward`)
- Uses NLI model (`microsoft/deberta-base-mnli`)
- Formula: `r_entailment = β * P(entailment) - α * P(contradiction)`
- Default parameters: α=5.0, β=10.0

#### Combined Reward (`CombinedReward`)
- Combines similarity and entailment rewards
- Formula: `r_combined = γ * r_similarity + r_entailment`
- Default parameters: γ=1.5, α=5.0, β=10.0

### 4. KGD Decoding Algorithm (`kgd_decode_single`)

Follows Algorithm 1 from the paper:

1. **Retrieve relevant knowledge** k = {k_i}_{i=1}^N from K based on query q
2. **Initialize generated tokens** x_{<1} with q prepended with contexts k
3. **For t = 1 to T do:**
   4. Compute language model logits z(x_t|x_{<t}) over vocabulary
   5. Select top-m tokens V_m based on logit values
   6. **For each x_t ∈ V_m do:**
      7. y_t = [x_{<t}, x_t]
      8. Compute knowledge-guided reward r(y_t, k)
   9. Update logits: z_KGD(x_t|x_{<t}) = z(x_t|x_{<t}) + w · r(y_t, k)
   10. **end for**
   11. Compute p_KGD(x_t|x_{<t}) by applying softmax over updated logits
   12. Sample next token x_t ~ p_KGD(x_t|x_{<t})
   13. Update generated tokens x_{<t+1} = [x_{<t}, x_t]
14. **end for**
15. **return** Generated text x_{<T+1}

### 5. Batch Processing (`kgd_decode`)

Processes multiple examples with configurable:
- Reward type: 'similarity', 'entailment', or 'combined'
- Weight parameters (w, α, β, γ)
- Top-m approximation (default m=4)
- Maximum tokens to generate

## Files Modified

1. **`src/decoding/kgd_decoder.py`** (complete rewrite)
   - Implements all KGD components
   - ~500 lines of well-documented code
   - Follows Algorithm 1 exactly

2. **`src/run_kgd.py`** (updated)
   - Tests all three reward types
   - Uses parameters from Section 5.3
   - Compares performance with different configurations

3. **`src/eval/metrics.py`** (updated)
   - Now supports both `VanillaPrediction` and `KGDPrediction`
   - Compatible with new API

4. **`requirements.txt`** (updated)
   - Added `sentence-transformers` dependency

## Usage Example

```python
from models.olmoe import OLMoELM
from data.datasets import load_nq
from decoding.kgd_decoder import kgd_decode
from eval.metrics import compute_em_f1

# Initialize model
model = OLMoELM()

# Load examples
examples = load_nq(split="validation[:50]")

# Similarity reward (w=2)
preds_sim = kgd_decode(
    model, 
    examples, 
    reward_type="similarity",
    weight=2.0,
    max_new_tokens=32,
    top_m=4
)

# Entailment reward (α=5, β=10)
preds_ent = kgd_decode(
    model,
    examples,
    reward_type="entailment",
    alpha=5.0,
    beta=10.0,
    max_new_tokens=32,
    top_m=4
)

# Combined reward (α=5, β=10, γ=1.5)
preds_combined = kgd_decode(
    model,
    examples,
    reward_type="combined",
    weight=1.5,  # γ (gamma) for similarity weight
    alpha=5.0,   # contradiction penalty
    beta=10.0,   # entailment reward
    max_new_tokens=32,
    top_m=4
)

# Evaluate
em, f1 = compute_em_f1(preds_sim)
print(f"Exact Match: {em:.2f}, F1: {f1:.2f}")
```

## Parameters from Section 5.3

### Short-form Generation
- Model: Llama 7B (Touvron et al., 2023a) and Llama 2 7B (Touvron et al., 2023b)
- Decoding strategies: greedy, beam search (width=4), contrastive search (k=4, penalty α=0.6)
- Top-m approximation: m=4

### Long-form Generation
- Contrastive search only
- Similarity reward: w=2
- Entailment reward: α=5, β=10
- Combined reward: α=5, β=10, γ=1.5

### Document Chunking
- Chunk size: 500 characters
- Overlap: 100 characters
- Retrieval model: `sentence-transformers/all-mpnet-base-v2`
- Top-N chunks: N=3

## Implementation Quality

✓ Follows Algorithm 1 exactly
✓ All parameters from Section 5.3 implemented
✓ Well-documented code with docstrings
✓ Type hints throughout
✓ No linter errors
✓ Syntactically valid Python
✓ Abstract base class for extensible reward functions
✓ Efficient batching and progress tracking
✓ Compatible with existing evaluation framework

## Testing

Run the implementation with:

```bash
cd /Users/elahehrasoulian/Documents/MSA_Program/LLM_CS8803/moe-knowledge-guided-decoding
python src/run_kgd.py
```

This will test all three reward types (similarity, entailment, combined) on a small validation set and report EM and F1 scores.

