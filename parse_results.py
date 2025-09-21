#!/usr/bin/env python3
"""
Parse CSDG vs KgCoOp experiment results and generate analysis
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def parse_log_file(log_path):
    """Parse training log to extract metrics"""
    with open(log_path, 'r') as f:
        content = f.read()

    # Extract final test results
    test_match = re.search(r'\* accuracy: ([0-9.]+)%', content)
    test_acc = float(test_match.group(1)) if test_match else None

    # Extract training metrics from last epoch
    last_epoch = re.findall(r'epoch \[10/10\].*?lr [0-9.e-]+', content, re.DOTALL)
    if not last_epoch:
        return {'test_accuracy': test_acc}

    last_log = last_epoch[-1]

    # Parse metrics
    metrics = {'test_accuracy': test_acc}

    patterns = {
        'alpha_mean': r'alpha_mean ([0-9.]+)',
        'alpha_std': r'alpha_std ([0-9.]+)',
        'gate_entropy': r'gate_entropy ([0-9.]+)',
        'loss_ce': r'loss_ce ([0-9.]+)',
        'loss_decor': r'loss_decor ([0-9.]+)',
        'loss_style_ce': r'loss_style_ce ([0-9.]+)',
        'train_acc': r'acc ([0-9.]+)'
    }

    for metric, pattern in patterns.items():
        match = re.search(pattern, last_log)
        metrics[metric] = float(match.group(1)) if match else None

    return metrics

def parse_all_experiments(results_dir="/content/drive/MyDrive/CSDG"):
    """Parse all experiment results"""
    results_dir = Path(results_dir)

    experiments = {
        'csdg_clipart_10ep': 'CSDG Baseline',
        'kgcoop_clipart_10ep': 'KgCoOp Baseline',
        'csdg_no_gate_10ep': 'CSDG No Gating',
        'csdg_no_decorr_10ep': 'CSDG No Decorr',
        'csdg_style_only_10ep': 'CSDG Style Only'
    }

    results = []

    for exp_dir, exp_name in experiments.items():
        exp_path = results_dir / exp_dir
        if not exp_path.exists():
            print(f"⚠️ Missing: {exp_dir}")
            continue

        # Find log file
        log_files = list(exp_path.glob("*.log")) + list(exp_path.glob("log.txt"))
        if not log_files:
            print(f"⚠️ No log file in: {exp_dir}")
            continue

        metrics = parse_log_file(log_files[0])
        metrics['experiment'] = exp_name
        metrics['exp_type'] = 'CSDG' if 'CSDG' in exp_name else 'KgCoOp'
        results.append(metrics)

        acc_str = f"{metrics['test_accuracy']:.2f}%" if metrics['test_accuracy'] else "No accuracy found"
        print(f"✅ Parsed: {exp_name} -> {acc_str}")

    return pd.DataFrame(results)

def create_visualizations(df):
    """Create useful plots for thesis"""

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CSDG vs KgCoOp: Experimental Analysis', fontsize=16, fontweight='bold')

    # 1. Test Accuracy Comparison
    ax = axes[0, 0]
    bars = ax.bar(range(len(df)), df['test_accuracy'],
                  color=['red' if 'KgCoOp' in exp else 'blue' for exp in df['experiment']])
    ax.set_title('Test Accuracy Comparison', fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['experiment'], rotation=45, ha='right')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Alpha Values (Gate Behavior)
    ax = axes[0, 1]
    csdg_data = df[df['exp_type'] == 'CSDG'].copy()
    if not csdg_data['alpha_mean'].isna().all():
        bars = ax.bar(csdg_data['experiment'], csdg_data['alpha_mean'])
        ax.set_title('Gate Behavior (α values)', fontweight='bold')
        ax.set_ylabel('α (0=style, 1=content)')
        ax.set_xticklabels(csdg_data['experiment'], rotation=45, ha='right')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Balanced')
        ax.legend()

    # 3. Performance vs Gate Mixing
    ax = axes[0, 2]
    csdg_with_alpha = csdg_data.dropna(subset=['alpha_mean'])
    if len(csdg_with_alpha) > 2:
        ax.scatter(csdg_with_alpha['alpha_mean'], csdg_with_alpha['test_accuracy'], s=100)
        ax.set_xlabel('α (Content Bias)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Performance vs Gate Bias', fontweight='bold')

        # Add labels
        for _, row in csdg_with_alpha.iterrows():
            ax.annotate(row['experiment'].replace('CSDG ', ''),
                       (row['alpha_mean'], row['test_accuracy']),
                       xytext=(5, 5), textcoords='offset points')

    # 4. Loss Components
    ax = axes[1, 0]
    loss_cols = ['loss_ce', 'loss_decor', 'loss_style_ce']
    csdg_losses = csdg_data[['experiment'] + loss_cols].set_index('experiment')
    csdg_losses = csdg_losses.dropna()

    if not csdg_losses.empty:
        csdg_losses.plot(kind='bar', ax=ax, stacked=True)
        ax.set_title('Loss Components', fontweight='bold')
        ax.set_ylabel('Loss Value')
        ax.legend(['CE Loss', 'Decorr Loss', 'Style CE Loss'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 5. CSDG Ablation Study
    ax = axes[1, 1]
    ablation_order = ['CSDG Baseline', 'CSDG No Gating', 'CSDG No Decorr', 'CSDG Style Only']
    ablation_data = df[df['experiment'].isin(ablation_order)]
    ablation_data = ablation_data.set_index('experiment').reindex(ablation_order)

    bars = ax.bar(range(len(ablation_data)), ablation_data['test_accuracy'],
                  color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax.set_title('CSDG Ablation Study', fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(range(len(ablation_data)))
    ax.set_xticklabels([exp.replace('CSDG ', '') for exp in ablation_data.index], rotation=45, ha='right')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 6. Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate key metrics
    kgcoop_acc = df[df['exp_type'] == 'KgCoOp']['test_accuracy'].iloc[0]
    best_csdg = df[df['exp_type'] == 'CSDG']['test_accuracy'].max()
    worst_csdg = df[df['exp_type'] == 'CSDG']['test_accuracy'].min()

    summary_text = f"""
    KEY FINDINGS:

    KgCoOp Baseline: {kgcoop_acc:.2f}%
    Best CSDG Variant: {best_csdg:.2f}%
    Worst CSDG Variant: {worst_csdg:.2f}%

    Gap to Beat: {kgcoop_acc - best_csdg:.2f}%

    INSIGHTS:
    • Gate learns to favor content
    • Style stream adds noise
    • No gating = best CSDG variant
    • Architecture issue, not training
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    return fig

def generate_thesis_tables(df):
    """Generate LaTeX-ready tables for thesis"""

    # Main results table
    results_table = df[['experiment', 'test_accuracy', 'alpha_mean']].copy()
    results_table.columns = ['Method', 'Accuracy (%)', 'α (Content Bias)']
    results_table['Accuracy (%)'] = results_table['Accuracy (%)'].round(2)
    results_table['α (Content Bias)'] = results_table['α (Content Bias)'].fillna('N/A')

    print("=" * 60)
    print("MAIN RESULTS TABLE (for thesis)")
    print("=" * 60)
    print(results_table.to_string(index=False))

    # Ablation analysis
    print("\n" + "=" * 60)
    print("ABLATION ANALYSIS")
    print("=" * 60)

    baseline_acc = df[df['experiment'] == 'CSDG Baseline']['test_accuracy'].iloc[0]
    kgcoop_acc = df[df['experiment'] == 'KgCoOp Baseline']['test_accuracy'].iloc[0]

    for _, row in df.iterrows():
        if 'CSDG' in row['experiment']:
            diff_from_baseline = row['test_accuracy'] - baseline_acc
            diff_from_kgcoop = row['test_accuracy'] - kgcoop_acc
            print(f"{row['experiment']:20} {row['test_accuracy']:6.2f}% "
                  f"({diff_from_baseline:+5.2f} vs baseline, {diff_from_kgcoop:+5.2f} vs KgCoOp)")

def main():
    print("🔍 Parsing CSDG vs KgCoOp experiment results...")

    # Parse results
    df = parse_all_experiments()

    if df.empty:
        print("❌ No results found!")
        return

    print(f"\n✅ Parsed {len(df)} experiments")

    # Generate visualizations
    print("📊 Creating visualizations...")
    fig = create_visualizations(df)

    # Save plots
    output_dir = Path("/content/drive/MyDrive/CSDG")
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / "experiment_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Generate tables
    generate_thesis_tables(df)

    # Save data
    df.to_csv(output_dir / "experiment_results.csv", index=False)
    print(f"\n💾 Results saved to {output_dir}")

    return df

if __name__ == "__main__":
    df = main()