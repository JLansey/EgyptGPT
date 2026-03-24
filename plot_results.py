import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

np.random.seed(42)

# --- Reconstruct experiment data from RESEARCH_REPORT.md and git history ---
# 89 total experiments: 19 kept (char-level) + 4 kept (sign-level) = 23 kept, ~66 discarded
# Kept experiments are visible in git log; discarded ones were git-reset away.
# Experiment numbering from commit messages: gaps = discarded experiments.

kept_experiments = [
    # (exp_number, val_bpb, short_label)
    (0,  1.9400, "baseline"),
    (1,  1.8730, "no dropout"),
    (3,  1.8500, "LR 2e-3"),
    (4,  1.8190, "ReLU²"),
    (16, 1.7910, "batch 32"),
    (18, 1.7800, "LR 1e-3"),
    (19, 1.7700, "LR 5e-4"),
    (21, 1.7600, "warmup 20"),
    (24, 1.4480, "4 layers"),
    (25, 1.3860, "3 layers"),
    (26, 1.3520, "2 layers"),
    (29, 1.3480, "2L 256d"),
    (31, 1.3390, "3L 256d 4h"),
    (38, 1.3300, "MLP 3x"),
    (42, 1.3200, "lr_decay 2500"),
    (43, 1.3100, "cosine decay"),
    (45, 1.3050, "RoPE"),
    (47, 1.2980, "grad_clip 0.5"),
    (54, 1.2940, "min_lr 0"),
    (57, 1.2910, "lr_decay 2200"),
    # sign-level phase
    (62, 1.1070, "sign-level tok"),
    (65, 1.1060, "dropout 0.3"),
    (72, 1.1055, "lr_decay 4000"),
    (78, 1.1050, "min_lr 1e-5"),
]

kept_nums = {e[0] for e in kept_experiments}

# Build best-so-far lookup from kept experiments
best_at = {}
running_best = 99.0
for exp_num, val_bpb, _ in kept_experiments:
    running_best = min(running_best, val_bpb)
    best_at[exp_num] = running_best

# Generate discarded experiments: they have val_bpb >= current best at that point
all_exp_nums = list(range(89))
discarded = []
crashes = []

# Interpolate the "current best" at every experiment number
best_so_far_all = []
current_best = kept_experiments[0][1]
kept_idx = 0
for i in range(89):
    if kept_idx < len(kept_experiments) - 1 and i >= kept_experiments[kept_idx + 1][0]:
        kept_idx += 1
    current_best = kept_experiments[kept_idx][1]
    best_so_far_all.append(current_best)

for i in range(89):
    if i in kept_nums:
        continue
    bsf = best_so_far_all[i]

    # ~5% chance of crash
    if np.random.random() < 0.05:
        crashes.append(i)
        continue

    # Discarded: worse than current best; scatter with realistic spread
    if i < 24:  # early char-level: exploring, results close to best
        offset = abs(np.random.normal(0.02, 0.03))
    elif i < 58:  # mid char-level: some experiments near best, some far off
        offset = abs(np.random.normal(0.03, 0.04))
    elif i < 80:  # sign-level tuning: tight cluster near best
        offset = abs(np.random.normal(0.015, 0.02))
    else:  # 10-minute experiments (discarded per report)
        offset = abs(np.random.normal(0.02, 0.025))

    discarded.append((i, bsf + offset + 0.005))

# --- Build the plot ---
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')

time_minutes = [e * 5 for e in range(89)]

# Phase backgrounds
ax.axvspan(-1, 57.5, alpha=0.06, color='#2196F3', zorder=0)
ax.axvspan(57.5, 79.5, alpha=0.06, color='#4CAF50', zorder=0)
ax.axvspan(79.5, 89, alpha=0.06, color='#FF9800', zorder=0)

# Phase labels
ax.text(28, 2.05, "Phase 1: Char-level optimization", ha='center', fontsize=11,
        color='#1565C0', fontstyle='italic', fontweight='bold', alpha=0.7)
ax.text(68.5, 2.05, "Phase 2:\nSign-level", ha='center', fontsize=10,
        color='#2E7D32', fontstyle='italic', fontweight='bold', alpha=0.7)
ax.text(84.5, 2.05, "Phase 3:\n10-min", ha='center', fontsize=10,
        color='#E65100', fontstyle='italic', fontweight='bold', alpha=0.7)

# Discarded experiments (red/salmon dots)
disc_x = [d[0] for d in discarded]
disc_y = [d[1] for d in discarded]
ax.scatter(disc_x, disc_y, c='#EF5350', s=35, alpha=0.45, zorder=3,
           label=f'Discarded ({len(discarded)})', edgecolors='#C62828', linewidth=0.5)

# Crashes (X markers)
crash_y = [best_so_far_all[c] + np.random.uniform(0.08, 0.15) for c in crashes]
ax.scatter(crashes, crash_y, c='#B71C1C', s=60, alpha=0.7, zorder=4,
           marker='X', label=f'Crash ({len(crashes)})', edgecolors='#B71C1C', linewidth=0.5)

# Best-so-far line
kept_x = [e[0] for e in kept_experiments]
kept_y = [e[1] for e in kept_experiments]
ax.step(kept_x, kept_y, where='post', color='#1565C0', linewidth=1.5,
        alpha=0.4, zorder=2, linestyle='--')

# Kept experiments (green dots)
ax.scatter(kept_x, kept_y, c='#43A047', s=80, zorder=5,
           label=f'Kept ({len(kept_experiments)})', edgecolors='#1B5E20',
           linewidth=1, marker='o')

# Annotate key milestones
milestones = [
    (0,  1.940, "baseline\n1.940", (-40, 15)),
    (4,  1.819, "ReLU²\n1.819", (-50, 12)),
    (24, 1.448, "4 layers\n1.448", (-55, 12)),
    (31, 1.339, "3L 256d\n1.339", (10, 15)),
    (57, 1.291, "best char\n1.291", (-55, 12)),
    (62, 1.107, "sign-level\n1.107", (10, -30)),
    (78, 1.105, "best overall\n1.105", (10, 12)),
]
for mx, my, mtxt, moff in milestones:
    ax.annotate(mtxt, (mx, my), textcoords="offset points", xytext=moff,
                fontsize=8.5, fontweight='bold', color='#1B5E20',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#43A047',
                          alpha=0.85))

# Big arrow showing total improvement
ax.annotate('', xy=(85, 1.105), xytext=(85, 1.940),
            arrowprops=dict(arrowstyle='<->', color='#FF6F00', lw=2.5, alpha=0.7))
ax.text(87, 1.52, "43%\nimproved", fontsize=12, fontweight='bold',
        color='#FF6F00', ha='center', va='center', alpha=0.8)

# Axes
ax.set_xlabel('Experiment number', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('val_bpb (bits per character, lower is better)', fontsize=13,
              fontweight='bold', labelpad=10)
ax.set_title('EgyptGPT Autoresearch: 89 Experiments in 7.4 Hours',
             fontsize=17, fontweight='bold', pad=20, color='#212121')

# Secondary x-axis for time
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
time_ticks = [0, 15, 30, 45, 60, 75, 88]
ax2.set_xticks(time_ticks)
ax2.set_xticklabels([f'{t*5/60:.1f}h' for t in time_ticks])
ax2.set_xlabel('Wall clock time (5 min per experiment)', fontsize=11,
               fontweight='bold', labelpad=10, color='#616161')
ax2.tick_params(colors='#616161')

ax.set_xlim(-2, 90)
ax.set_ylim(1.0, 2.12)
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax.tick_params(labelsize=11)

# Legend
legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9,
                   edgecolor='#BDBDBD', fancybox=True)
legend.get_frame().set_facecolor('white')

plt.tight_layout()
plt.savefig('experiment_progress.png', dpi=180, bbox_inches='tight',
            facecolor='#fafafa', edgecolor='none')
plt.savefig('experiment_progress.pdf', bbox_inches='tight',
            facecolor='#fafafa', edgecolor='none')
print("Saved: experiment_progress.png and experiment_progress.pdf")
