from models.olmoe import OLMoELM
from data.datasets import load_nq
from decoding.kgd_decoder import kgd_decode
from eval.metrics import compute_em_f1


def main():
    model = OLMoELM()

    # small slice first so it's fast
    examples = load_nq(split="train[:20]")

    # try a couple of alpha values later
    alpha = 0.5
    preds = kgd_decode(model, examples, alpha=alpha)

    em, f1 = compute_em_f1(preds)
    print(f"KGD (alpha={alpha})")
    print(f"Exact Match: {em:.2f}")
    print(f"F1         : {f1:.2f}")


if __name__ == "__main__":
    main()
