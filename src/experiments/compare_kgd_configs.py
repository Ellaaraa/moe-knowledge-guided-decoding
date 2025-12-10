"""
Compare different KGD configurations against vanilla baseline.

Usage:
    python src/experiments/compare_kgd_configs.py --num-examples 50
    
On Colab:
    !python src/experiments/compare_kgd_configs.py --num-examples 50

Output:
    - Console summary table
    - JSON results file in results/
    - Bar chart comparison (PNG)
    - Delta improvement chart (PNG)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.olmoe import OLMoELM
from data.datasets import load_nq
from decoding.kgd_decoder import kgd_decode
from decoding.vanilla import decode_vanilla
from eval.metrics import compute_em_f1


def run_experiment(
    model: OLMoELM,
    examples: List,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run a single experiment configuration."""
    
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    if config["type"] == "vanilla":
        preds = decode_vanilla(
            model, 
            examples, 
            max_new_tokens=config.get("max_new_tokens", 32),
        )
    else:
        preds = kgd_decode(
            model,
            examples,
            reward_type=config["reward_type"],
            weight=config.get("weight", 2.0),
            alpha=config.get("alpha", 5.0),
            beta=config.get("beta", 10.0),
            top_m=config.get("top_m", 4),
            max_new_tokens=config.get("max_new_tokens", 32),
        )
    
    em, f1 = compute_em_f1(preds)
    print(f"  EM: {em:.2f}, F1: {f1:.2f}")
    
    # Clear GPU cache to prevent memory accumulation between configurations
    torch.cuda.empty_cache()
    
    return {"EM": em, "F1": f1}


def plot_results(results: Dict[str, Any], output_dir: str, timestamp: str):
    """Generate visualization charts for the results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
        return
    
    # Extract data
    configs = list(results.keys())
    ems = [results[c]["metrics"]["EM"] for c in configs]
    f1s = [results[c]["metrics"]["F1"] for c in configs]
    
    vanilla_em = results["Vanilla"]["metrics"]["EM"]
    vanilla_f1 = results["Vanilla"]["metrics"]["F1"]
    
    delta_ems = [em - vanilla_em for em in ems]
    delta_f1s = [f1 - vanilla_f1 for f1 in f1s]
    
    # Set style - use a style that's likely available
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')
    
    # ============ Chart 1: Absolute Scores ============
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ems, width, label='Exact Match (EM)', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color='#A23B72', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars1, ems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('KGD Configuration Comparison: Absolute Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(ems), max(f1s)) * 1.15)
    
    # Add horizontal line for vanilla baseline
    ax.axhline(y=vanilla_em, color='#2E86AB', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=vanilla_f1, color='#A23B72', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, f'kgd_comparison_absolute_{timestamp}.png')
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart1_path}")
    plt.close()
    
    # ============ Chart 2: Delta from Vanilla ============
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Skip vanilla (delta is 0)
    kgd_configs = [c for c in configs if c != "Vanilla"]
    kgd_delta_ems = [delta_ems[i] for i, c in enumerate(configs) if c != "Vanilla"]
    kgd_delta_f1s = [delta_f1s[i] for i, c in enumerate(configs) if c != "Vanilla"]
    
    if kgd_configs:  # Only plot if there are KGD configs
        x = np.arange(len(kgd_configs))
        
        bars1 = ax.bar(x - width/2, kgd_delta_ems, width, label='Δ EM', 
                       color=['#28A745' if d >= 0 else '#DC3545' for d in kgd_delta_ems],
                       edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, kgd_delta_f1s, width, label='Δ F1',
                       color=['#20C997' if d >= 0 else '#E83E8C' for d in kgd_delta_f1s],
                       edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars1, kgd_delta_ems):
            ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:+.1f}', ha='center', va=va, fontsize=8)
        for bar, val in zip(bars2, kgd_delta_f1s):
            ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:+.1f}', ha='center', va=va, fontsize=8)
        
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Δ Score (vs Vanilla)', fontsize=12)
        ax.set_title('KGD Configuration Comparison: Improvement over Vanilla', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(kgd_configs, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, f'kgd_comparison_delta_{timestamp}.png')
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart2_path}")
    plt.close()
    
    # ============ Chart 3: Grouped by Reward Type ============
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Group configs by reward type
    groups = {
        'Similarity': [c for c in configs if 'Sim' in c],
        'Entailment': [c for c in configs if 'Ent' in c],
        'Combined': [c for c in configs if 'Comb' in c],
    }
    
    for ax, (group_name, group_configs) in zip(axes, groups.items()):
        if not group_configs:
            ax.text(0.5, 0.5, 'No configs', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(group_name)
            continue
            
        group_ems = [results[c]["metrics"]["EM"] for c in group_configs]
        group_f1s = [results[c]["metrics"]["F1"] for c in group_configs]
        
        x = np.arange(len(group_configs))
        bars1 = ax.bar(x - width/2, group_ems, width, label='EM', color='#2E86AB')
        bars2 = ax.bar(x + width/2, group_f1s, width, label='F1', color='#A23B72')
        
        ax.axhline(y=vanilla_em, color='#2E86AB', linestyle='--', alpha=0.5, label='Vanilla EM')
        ax.axhline(y=vanilla_f1, color='#A23B72', linestyle='--', alpha=0.5, label='Vanilla F1')
        
        ax.set_title(f'{group_name} Reward', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        # Simplify labels by removing prefix
        simplified_labels = []
        for c in group_configs:
            label = c.replace('Sim_', '').replace('Ent_', '').replace('Comb_', '')
            simplified_labels.append(label)
        ax.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score (%)')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, max(max(ems), max(f1s)) * 1.15)
    
    plt.suptitle('KGD Performance by Reward Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, f'kgd_comparison_by_type_{timestamp}.png')
    plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart3_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare KGD configurations")
    parser.add_argument(
        "--num-examples", "-n", type=int, default=10,
        help="Number of examples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--max-tokens", "-t", type=int, default=32,
        help="Max tokens to generate (default: 32)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer configs, 10 examples"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Use 4-bit quantization to reduce memory usage (recommended for Colab free tier)"
    )
    args = parser.parse_args()
    
    if args.quick:
        args.num_examples = 10
    
    # Define experiment configurations
    configs = [
        # Baseline
        {"name": "Vanilla", "type": "vanilla", "max_new_tokens": args.max_tokens},
        
        # Similarity reward experiments
        {"name": "Sim_w1.0", "type": "kgd", "reward_type": "similarity", 
         "weight": 1.0, "top_m": 4, "max_new_tokens": args.max_tokens},
        {"name": "Sim_w2.0", "type": "kgd", "reward_type": "similarity", 
         "weight": 2.0, "top_m": 4, "max_new_tokens": args.max_tokens},
        {"name": "Sim_w5.0", "type": "kgd", "reward_type": "similarity", 
         "weight": 5.0, "top_m": 4, "max_new_tokens": args.max_tokens},
        {"name": "Sim_w2_m8", "type": "kgd", "reward_type": "similarity", 
         "weight": 2.0, "top_m": 8, "max_new_tokens": args.max_tokens},
    ]
    
    # Add more configs unless quick mode
    if not args.quick:
        configs.extend([
            # Entailment experiments
            {"name": "Ent_default", "type": "kgd", "reward_type": "entailment",
             "alpha": 5.0, "beta": 10.0, "top_m": 4, "max_new_tokens": args.max_tokens},
            {"name": "Ent_strong", "type": "kgd", "reward_type": "entailment",
             "alpha": 10.0, "beta": 20.0, "top_m": 4, "max_new_tokens": args.max_tokens},
            
            # Combined experiments
            {"name": "Comb_g1.5", "type": "kgd", "reward_type": "combined",
             "weight": 1.5, "alpha": 5.0, "beta": 10.0, "top_m": 4, 
             "max_new_tokens": args.max_tokens},
            {"name": "Comb_g3.0", "type": "kgd", "reward_type": "combined",
             "weight": 3.0, "alpha": 5.0, "beta": 10.0, "top_m": 4,
             "max_new_tokens": args.max_tokens},
        ])
    
    print("="*60)
    print("KGD CONFIGURATION COMPARISON")
    print("="*60)
    print(f"Examples: {args.num_examples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Configurations: {len(configs)}")
    
    # Load model
    print("\nLoading model...")
    if args.use_4bit:
        print("Using 4-bit quantization to reduce memory usage")
    model = OLMoELM(use_4bit=args.use_4bit)
    print(f"Model on: {model.device}")
    
    # Load data
    print(f"\nLoading {args.num_examples} examples...")
    examples = load_nq(split="validation", config="dev", max_examples=args.num_examples)
    print(f"Loaded {len(examples)} examples")
    
    # Run experiments
    results = {}
    for config in configs:
        metrics = run_experiment(model, examples, config)
        results[config["name"]] = {
            "config": config,
            "metrics": metrics,
        }
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<20} {'EM':>10} {'F1':>10} {'Δ EM':>10} {'Δ F1':>10}")
    print("-"*60)
    
    vanilla_em = results["Vanilla"]["metrics"]["EM"]
    vanilla_f1 = results["Vanilla"]["metrics"]["F1"]
    
    for name, data in results.items():
        em = data["metrics"]["EM"]
        f1 = data["metrics"]["F1"]
        delta_em = em - vanilla_em
        delta_f1 = f1 - vanilla_f1
        delta_em_str = f"+{delta_em:.2f}" if delta_em >= 0 else f"{delta_em:.2f}"
        delta_f1_str = f"+{delta_f1:.2f}" if delta_f1 >= 0 else f"{delta_f1:.2f}"
        print(f"{name:<20} {em:>10.2f} {f1:>10.2f} {delta_em_str:>10} {delta_f1_str:>10}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = os.path.join(args.output_dir, f'kgd_comparison_{timestamp}.json')
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_examples": args.num_examples,
            "max_tokens": args.max_tokens,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate plots
    if not args.no_plot:
        print("\nGenerating visualizations...")
        plot_results(results, args.output_dir, timestamp)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

