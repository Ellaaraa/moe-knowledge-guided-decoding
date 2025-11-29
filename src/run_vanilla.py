from models.olmoe import OLMoELM
from data.datasets import load_nq
from decoding.vanilla import decode_vanilla
from eval.metrics import compute_em_f1


def main():
    model = OLMoELM()

    examples = load_nq(
        split="validation",   # or "train"
        config="dev",         # or "default" depending on which you want
        max_examples=50,      # keep small in Colab at first
    )

    preds = decode_vanilla(model, examples)

    em, f1 = compute_em_f1(preds)
    print(f"Exact Match: {em:.2f}")
    print(f"F1         : {f1:.2f}")


if __name__ == "__main__":
    main()
