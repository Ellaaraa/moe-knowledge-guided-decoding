from models.llama import CausalLM
from data.datasets import load_nq
from decoding.kgd_decoder import kgd_decode
from eval.metrics import compute_em_f1


def main():
    # Using Qwen2.5-3B-Instruct (instruction-tuned, smaller, no license gate)
    model = CausalLM("Qwen/Qwen2.5-3B-Instruct")

    # small slice first so it's fast
    examples = load_nq(split="validation[:50]")

    # Test different reward types with parameters from Section 5.3
    
    # 1. Similarity reward (w=2)
    print("\n" + "="*50)
    print("KGD with Similarity Reward")
    print("="*50)
    preds_sim = kgd_decode(
        model, 
        examples, 
        reward_type="similarity",
        weight=2.0,
        max_new_tokens=32,
        top_m=10
    )
    em, f1 = compute_em_f1(preds_sim)
    print(f"Exact Match: {em:.2f}")
    print(f"F1         : {f1:.2f}")

    # 2. Entailment reward (α=5, β=10)
    print("\n" + "="*50)
    print("KGD with Entailment Reward")
    print("="*50)
    preds_ent = kgd_decode(
        model,
        examples,
        reward_type="entailment",
        alpha=5.0,
        beta=10.0,
        max_new_tokens=32,
        top_m=10
    )
    em, f1 = compute_em_f1(preds_ent)
    print(f"Exact Match: {em:.2f}")
    print(f"F1         : {f1:.2f}")

    # 3. Combined reward (α=5, β=10, γ=1.5)
    print("\n" + "="*50)
    print("KGD with Combined Reward")
    print("="*50)
    preds_combined = kgd_decode(
        model,
        examples,
        reward_type="combined",
        weight=1.5,  # γ (gamma) for similarity weight
        alpha=5.0,   # contradiction penalty
        beta=10.0,   # entailment reward
        max_new_tokens=32,
        top_m=10
    )
    em, f1 = compute_em_f1(preds_combined)
    print(f"Exact Match: {em:.2f}")
    print(f"F1         : {f1:.2f}")
    
    return preds_sim, preds_ent, preds_combined


if __name__ == "__main__":
    main()
