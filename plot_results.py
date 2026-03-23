#!/usr/bin/env python3
"""Generate comparison charts for MG-GAT reproduction report."""

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Fig 1: Experiment progression — Val & Test RMSE across experiments
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6.5))

exps = ['Exp0\nBaseline', 'Exp2\nStrong reg', 'Exp3\nSmall model', 'Exp4\nSmall+strong', 'Final\nHyperopt']
val_rmse  = [1.4189, 1.3656, 1.4034, 1.3735, 1.3619]
test_rmse = [1.4593, 1.4098, 1.4449, 1.4162, 1.4096]

# Parameter annotations for each experiment
param_text = [
    r'$\theta_1$=1e-3, $\theta_2$=1e-3' + '\nkf=64, d0=64, d1=128\nlr=1e-3, actv1=elu',
    r'$\theta_1$=0.1, $\theta_2$=0.01' + '\nkf=64, d0=64, d1=128\nlr=1e-3, actv1=elu',
    r'$\theta_1$=0.01, $\theta_2$=0.01' + '\nkf=32, d0=32, d1=64\nlr=1e-3, actv1=elu',
    r'$\theta_1$=0.1, $\theta_2$=0.01' + '\nkf=32, d0=32, d1=64\nlr=1e-3, actv1=elu',
    r'$\theta_1$=0.1, $\theta_2$=0.1' + '\nkf=128, d0=64, d1=64\nlr=5e-3, actv1=relu',
]

x = np.arange(len(exps))
w = 0.32
bars1 = ax.bar(x - w/2, val_rmse, w, label='Val RMSE', color='#4C72B0', edgecolor='white')
bars2 = ax.bar(x + w/2, test_rmse, w, label='Test RMSE', color='#DD8452', edgecolor='white')

ax.axhline(y=1.249, color='red', linestyle='--', linewidth=1.5, label='Paper MG-GAT (1.249)')
ax.set_xticks(x)
ax.set_xticklabels(exps, fontsize=10)
ax.set_ylabel('RMSE')
ax.set_title('Experiment Progression: Val & Test RMSE')
ax.set_ylim(1.17, 1.52)
ax.legend(loc='upper right')
ax.bar_label(bars1, fmt='%.4f', fontsize=8, padding=2)
ax.bar_label(bars2, fmt='%.4f', fontsize=8, padding=2)
ax.grid(axis='y', alpha=0.3)

# Add parameter text below each group
for i, txt in enumerate(param_text):
    ax.text(x[i], 1.185, txt, ha='center', va='top', fontsize=7,
            fontstyle='italic', color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='#CCCCCC', alpha=0.9))

fig.savefig(os.path.join(out_dir, 'fig1_experiment_progression.png'))
plt.close(fig)

# ============================================================
# Fig 2: Comparison with paper baselines
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

models = ['MG-GAT\n(Paper)', 'DGAN\n(Paper)', 'GRALS\n(Paper)', 'SVD++\n(Paper)', 'MG-GAT\n(Ours)']
rmse_vals = [1.249, 1.250, 1.328, 1.339, 1.4096]
colors = ['#55A868', '#55A868', '#55A868', '#55A868', '#DD8452']

bars = ax.barh(models, rmse_vals, color=colors, edgecolor='white', height=0.55)
ax.set_xlabel('Test RMSE')
ax.set_title('Comparison with Paper Baselines')
ax.set_xlim(1.15, 1.50)
ax.invert_yaxis()

for bar, v in zip(bars, rmse_vals):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.4f}',
            va='center', fontsize=11, fontweight='bold')

ax.grid(axis='x', alpha=0.3)

# Warning: different datasets
ax.text(0.5, -0.1,
        'Note: Paper baselines use 2019 Yelp PA (77K users); Ours uses 2024 version (320K users). Not directly comparable.',
        ha='center', fontsize=8, fontstyle='italic', color='#C44E52',
        transform=ax.transAxes)

fig.savefig(os.path.join(out_dir, 'fig2_baseline_comparison.png'))
plt.close(fig)

# ============================================================
# Fig 3: Train-Test gap reduction
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

exps_short = ['Exp0', 'Exp2', 'Exp3', 'Exp4', 'Final']
gaps = [0.33, 0.31, 0.32, 0.30, 0.40]

# Key change annotations
change_notes = [
    'Baseline\n' + r'$\theta_1$=1e-3, $\theta_2$=1e-3' + '\nkf=64, d0=64, d1=128',
    'Stronger reg\n' + r'$\theta_1$=0.1, $\theta_2$=0.01',
    'Smaller model\nkf=32, d0=32, d1=64',
    'Small+strong reg\n' + r'$\theta_1$=0.1, $\theta_2$=0.01',
    'Hyperopt best\nlr=5e-3, actv1=relu',
]

test_rmse_gap = [1.4593, 1.4098, 1.4449, 1.4162, 1.4096]

ax.plot(exps_short, gaps, 'o-', color='#C44E52', linewidth=2.5, markersize=10, zorder=3)
for i, g in enumerate(gaps):
    ax.annotate(f'Gap {g:.2f}\nTest {test_rmse_gap[i]:.4f}', (exps_short[i], g),
                textcoords="offset points",
                xytext=(0, 14), ha='center', fontsize=10, fontweight='bold')
    # Parameter note below the line
    ax.annotate(change_notes[i], (exps_short[i], g), textcoords="offset points",
                xytext=(0, -60), ha='center', fontsize=8, fontstyle='italic', color='#444444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='#CCCCCC', alpha=0.9))

ax.set_ylabel('Train–Test RMSE Gap')
ax.set_title('Train–Test Gap Across Experiments')
ax.set_ylim(0.15, 0.48)
ax.grid(alpha=0.3)

# Explanation for Final's higher gap
ax.text(0.5, -0.08,
        'Note: Final has the highest gap but the best Test RMSE (1.4096) — higher lr allows deeper training fit.',
        ha='center', fontsize=8, fontstyle='italic', color='#666666',
        transform=ax.transAxes)
fig.savefig(os.path.join(out_dir, 'fig3_gap_reduction.png'))
plt.close(fig)

# ============================================================
# Fig 4: Ablation study — Ours vs Paper (all 6 configs)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

configs = ['Full\nMG-GAT', 'NIG\nRemoved', 'FR\nRemoved', 'Uniform\nGraph Wt', 'No Aux\nInfo', 'No Net\n+ No Aux']
ours_test = [1.4096, 1.4028, 1.4111, 1.4042, 1.4390, 1.4640]
paper_test = [1.249, 1.303, 1.305, 1.280, 1.312, 1.405]
colors6 = ['#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#DD8452', '#C44E52']

# Left: our ablation
bars = axes[0].bar(configs, ours_test, color=colors6, edgecolor='white', width=0.6)
axes[0].set_ylabel('Test RMSE')
axes[0].set_title('Ablation Study (Ours)')
axes[0].set_ylim(1.38, 1.50)
axes[0].bar_label(bars, fmt='%.4f', fontsize=8, fontweight='bold', padding=3)
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', labelsize=8)

# Right: paper ablation
bars = axes[1].bar(configs, paper_test, color=colors6, edgecolor='white', width=0.6)
axes[1].set_ylabel('Test RMSE')
axes[1].set_title('Ablation Study (Paper)')
axes[1].set_ylim(1.20, 1.45)
axes[1].bar_label(bars, fmt='%.3f', fontsize=8, fontweight='bold', padding=3)
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', labelsize=8)

fig.suptitle('Ablation Study: Full Table 2 Comparison', fontsize=14, fontweight='bold', y=1.02)
fig.text(0.5, -0.02,
         r'Base params: $\theta_1$=0.1, $\theta_2$=0.1, kf=128, d0=64, d1=64, lr=5e-3, actv1=relu',
         ha='center', fontsize=9, fontstyle='italic', color='#666666')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig4_ablation_comparison.png'))
plt.close(fig)

# ============================================================
# Fig 5: Dataset comparison (Ours vs Paper)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: counts
categories = ['Users', 'Businesses', 'Reviews']
ours_counts = [320212, 31663, 1183126]
paper_counts = [76865, 10966, 260350]

x = np.arange(len(categories))
w = 0.32
b1 = axes[0].bar(x - w/2, ours_counts, w, label='This Work', color='#DD8452', edgecolor='white')
b2 = axes[0].bar(x + w/2, paper_counts, w, label='Paper', color='#55A868', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].set_ylabel('Count')
axes[0].set_title('Dataset Scale')
axes[0].legend()
axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
axes[0].grid(axis='y', alpha=0.3)

# Right: ratios
ratio_labels = ['Users\n(×4.2)', 'Businesses\n(×2.9)', 'Reviews\n(×4.5)']
ratios = [320212/76865, 31663/10966, 1183126/260350]

bars = axes[1].bar(ratio_labels, ratios, color=['#C44E52', '#C44E52', '#C44E52'], edgecolor='white', width=0.45)
axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
axes[1].set_ylabel('Ratio (Ours / Paper)')
axes[1].set_title('Dataset Size Ratio')
axes[1].bar_label(bars, fmt='%.1f×', fontsize=12, fontweight='bold', padding=3)
axes[1].grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_dataset_comparison.png'))
plt.close(fig)

# ============================================================
# Fig 6: User filtering ablation
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

configs = ['Full Data\n(820K reviews)\n320K users', 'Filtered ≥2\n(698K reviews)\n95K users']
val_f = [1.3619, 1.3729]
test_f = [1.4096, 1.4193]

x = np.arange(len(configs))
w = 0.3
b1 = ax.bar(x - w/2, val_f, w, label='Val RMSE', color='#4C72B0', edgecolor='white')
b2 = ax.bar(x + w/2, test_f, w, label='Test RMSE', color='#DD8452', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel('RMSE')
ax.set_title('User Filtering Ablation')
ax.set_ylim(1.34, 1.44)
ax.legend()
ax.bar_label(b1, fmt='%.4f', fontsize=10, fontweight='bold', padding=3)
ax.bar_label(b2, fmt='%.4f', fontsize=10, fontweight='bold', padding=3)
ax.grid(axis='y', alpha=0.3)

# Add shared params note
ax.text(0.5, 1.345, r'Params: $\theta_1$=0.1, $\theta_2$=0.01, kf=32, d0=32, d1=64, lr=5e-3, actv1=tanh',
        ha='center', fontsize=8, fontstyle='italic', color='#666666',
        transform=ax.get_xaxis_transform())

fig.savefig(os.path.join(out_dir, 'fig6_user_filtering.png'))
plt.close(fig)

# ============================================================
# Fig 7: Gap analysis breakdown (4 factors)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

total_gap = 1.4096 - 1.249  # 0.1606

factors = [
    'Dataset sparsity\n(70% users have 1 rating)',
    'Missing LLM\nperceptual map',
    'Hyperparameter\nsearch (~3% coverage)',
    'Random seed /\ninitialisation',
]
estimates = [0.09, 0.025, 0.025, 0.015]  # midpoints of ranges
ranges_lo = [0.08, 0.02, 0.02, 0.01]
ranges_hi = [0.10, 0.03, 0.03, 0.02]

# Stacked horizontal bar
left = 0
colors_gap = ['#C44E52', '#8172B2', '#CCB974', '#64B5CD']
for i, (f, e) in enumerate(zip(factors, estimates)):
    bar = ax.barh(0, e, left=left, color=colors_gap[i], edgecolor='white', height=0.45,
                  label=f'{f}: ~{ranges_lo[i]:.2f}–{ranges_hi[i]:.2f}')
    ax.text(left + e/2, 0, f'{ranges_lo[i]:.2f}–{ranges_hi[i]:.2f}',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    left += e

# Mark total gap
ax.axvline(x=total_gap, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(total_gap + 0.002, 0.25, f'Actual gap\n= {total_gap:.3f}', fontsize=9, va='center')

ax.set_xlim(0, total_gap + 0.04)
ax.set_yticks([])
ax.set_xlabel('RMSE Gap Contribution')
ax.set_title(f'Gap Decomposition: 1.4096 (Ours) − 1.249 (Paper) = {total_gap:.3f}')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# Add summary text
ax.text(0.5, -0.15, f'Total estimated range: ~{sum(ranges_lo):.2f}–{sum(ranges_hi):.2f}  |  Observed gap: {total_gap:.3f}',
        ha='center', fontsize=10, fontstyle='italic', color='#666666',
        transform=ax.transAxes)

fig.savefig(os.path.join(out_dir, 'fig7_gap_analysis.png'))
plt.close(fig)

print(f"All figures saved to {out_dir}/")
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
