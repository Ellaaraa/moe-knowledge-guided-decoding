"""
Inspect KGD vs Vanilla decoding trajectories step by step.

This script runs KGD decoding on a few examples and prints:
- At each step: the vanilla next token vs the KGD next token
- The top-m candidate tokens and their rewards
- Partial decoded strings at each step
- Final answers from both methods

Usage:
    cd /path/to/moe-knowledge-guided-decoding
    python -m src.eval.inspect_trajectory
"""

from __future__ import annotations

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.olmoe import OLMoELM
from data.datasets import load_nq
from decoding.kgd_decoder import (
    SimilarityReward,
    EntailmentReward,
    CombinedReward,
    debug_trajectory,
)


def print_trajectory(info: dict, show_top_m: bool = True, show_partials: bool = True):
    """Pretty-print a trajectory comparison."""
    
    print("\n" + "=" * 80)
    print(f"QUESTION: {info['question']}")
    print(f"GOLD ANSWERS: {info['gold_answers']}")
    print(f"Reward type: {info['reward_type']}, weight={info['weight']}")
    print("-" * 80)
    
    # Show retrieved knowledge (truncated)
    if "knowledge_chunks" in info and info["knowledge_chunks"]:
        print("\nRETRIEVED KNOWLEDGE (truncated):")
        for i, chunk in enumerate(info["knowledge_chunks"][:2]):
            truncated = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(f"  [{i+1}] {truncated}")
    
    print("\n" + "-" * 80)
    print("STEP-BY-STEP COMPARISON:")
    print("-" * 80)
    
    if not info["steps"]:
        print("  (No steps - fallback to vanilla or no context)")
        return
    
    # Header
    print(f"{'Step':>4}  {'Match':>5}  {'Vanilla Token':>20}  {'KGD Token':>20}")
    print("-" * 60)
    
    for step in info["steps"]:
        match_str = "YES" if step["tokens_match"] else "NO"
        vanilla_tok = repr(step["vanilla_token"])[:18]
        kgd_tok = repr(step["kgd_token"])[:18]
        
        print(f"{step['t']:>4}  {match_str:>5}  {vanilla_tok:>20}  {kgd_tok:>20}")
        
        if show_top_m:
            # Show top-m candidates with rewards
            print("        Top-m candidates:")
            for tid, tok, logit, reward in zip(
                step["top_m_ids"],
                step["top_m_tokens"],
                step["top_m_logits"],
                step["rewards"],
            ):
                marker = " <--" if tid == step["kgd_token_id"] else ""
                print(
                    f"          id={tid:6d}  token={repr(tok):15s}  "
                    f"logit={logit:7.2f}  reward={reward:.4f}{marker}"
                )
        
        if show_partials and step["t"] % 5 == 4:  # Show partial every 5 steps
            print(f"        [Partial vanilla]: {repr(step['partial_vanilla'][:60])}")
            print(f"        [Partial KGD]:     {repr(step['partial_kgd'][:60])}")
    
    print("\n" + "-" * 80)
    print("FINAL RESULTS:")
    print("-" * 80)
    print(f"  Steps: {info['num_steps']}")
    print(f"  Matching steps: {info['num_matching_steps']} / {info['num_steps']} "
          f"({100*info['num_matching_steps']/max(info['num_steps'],1):.1f}%)")
    print(f"\n  VANILLA ANSWER: {repr(info['final_vanilla_answer'])}")
    print(f"  KGD ANSWER:     {repr(info['final_kgd_answer'])}")
    print(f"  GOLD ANSWERS:   {info['gold_answers']}")
    
    # Check if either answer matches gold
    vanilla_match = any(
        gold.lower() in info["final_vanilla_answer"].lower() 
        for gold in info["gold_answers"]
    )
    kgd_match = any(
        gold.lower() in info["final_kgd_answer"].lower() 
        for gold in info["gold_answers"]
    )
    print(f"\n  Vanilla contains gold? {vanilla_match}")
    print(f"  KGD contains gold?     {kgd_match}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect KGD vs Vanilla trajectories")
    parser.add_argument(
        "--num-examples", "-n", type=int, default=2,
        help="Number of examples to inspect (default: 2)"
    )
    parser.add_argument(
        "--max-tokens", "-t", type=int, default=16,
        help="Maximum tokens to generate (default: 16)"
    )
    parser.add_argument(
        "--reward", "-r", type=str, default="similarity",
        choices=["similarity", "entailment", "combined"],
        help="Reward type (default: similarity)"
    )
    parser.add_argument(
        "--weight", "-w", type=float, default=2.0,
        help="Reward weight (default: 2.0)"
    )
    parser.add_argument(
        "--top-m", "-m", type=int, default=4,
        help="Top-m tokens to consider (default: 4)"
    )
    parser.add_argument(
        "--no-top-m", action="store_true",
        help="Don't show top-m candidate details"
    )
    parser.add_argument(
        "--no-partials", action="store_true",
        help="Don't show partial decoded strings"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output during decoding"
    )
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Use 4-bit quantization to reduce memory usage (recommended for Colab)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KGD vs VANILLA TRAJECTORY INSPECTOR")
    print("=" * 80)
    
    # Load model
    print("\nLoading OLMoE model...")
    if args.use_4bit:
        print("Using 4-bit quantization to reduce memory usage")
    model = OLMoELM(use_4bit=args.use_4bit)
    print(f"Model loaded on device: {model.device}")
    
    # Load examples
    print(f"\nLoading {args.num_examples} examples from Natural Questions...")
    examples = load_nq(split="validation", config="dev", max_examples=args.num_examples)
    print(f"Loaded {len(examples)} examples")
    
    # Initialize reward function
    print(f"\nUsing reward type: {args.reward}, weight: {args.weight}")
    if args.reward == "similarity":
        reward_fn = SimilarityReward()
    elif args.reward == "entailment":
        reward_fn = EntailmentReward(alpha=5.0, beta=10.0)
    elif args.reward == "combined":
        reward_fn = CombinedReward(similarity_weight=args.weight, alpha=5.0, beta=10.0)
    else:
        raise ValueError(f"Unknown reward type: {args.reward}")
    
    # Process each example
    for i, ex in enumerate(examples):
        print(f"\n\n{'#' * 80}")
        print(f"# EXAMPLE {i+1} / {len(examples)}")
        print(f"{'#' * 80}")
        
        info = debug_trajectory(
            model=model,
            example=ex,
            reward_function=reward_fn,
            weight=args.weight,
            max_new_tokens=args.max_tokens,
            top_m=args.top_m,
            temperature=0.0,
            verbose=args.verbose,
        )
        
        print_trajectory(
            info, 
            show_top_m=not args.no_top_m,
            show_partials=not args.no_partials,
        )
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

