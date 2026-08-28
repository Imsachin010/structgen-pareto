"""
figure3_alignment_repair.py
Two-panel figure for §4 Results, Finding 2 + 3.
- Left:  Alignment score drift across Stage 1→2→3
- Right: Repair breakdown for nested schema (N=30 queries)

Run in Colab or locally. Saves figure3_alignment_repair.pdf / .png
"""

import matplotlib
matplotlib.use("Agg")          # remove this line if running in Colab / Jupyter
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "experiments", "analysis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ── Left panel: alignment score drift ─────────────────────────────────────
stages_align = ['Stage 1\nEmbedding', 'Stage 2\nEmbedding\n+VLM', 'Stage 3\nTF-IDF']
alignment    = [0.938, 0.932, 0.899]

ax1.plot(stages_align, alignment,
         marker='o', color='#534AB7', linewidth=2, markersize=7)
ax1.set_ylim(0.85, 0.96)
ax1.set_ylabel('Query-Object Alignment Score', fontsize=10)
ax1.set_title('Alignment Degradation by Stage', fontsize=11, fontweight='normal')
for x, y in enumerate(alignment):
    ax1.annotate(f'{y:.3f}', (x, y),
                 textcoords='offset points', xytext=(0, 8),
                 ha='center', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Right panel: repair breakdown for nested schema ────────────────────────
categories  = ['JSON Parse\nRepairs', 'Schema\nRepairs', 'Unrecoverable']
values      = [18, 7, 8]
bar_colors  = ['#EF9F27', '#534AB7', '#E24B4A']

bars2 = ax2.bar(categories, values,
                color=bar_colors, width=0.5, edgecolor='white')
ax2.set_ylabel('Count (N=30 queries)', fontsize=10)
ax2.set_title('Repair Loop Activation (Complex Schema)',
              fontsize=11, fontweight='normal')
for bar, val in zip(bars2, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3, str(val),
             ha='center', fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

# Save both PDF and PNG
out_base = os.path.join(OUT_DIR, "figure3_alignment_repair")
plt.savefig(out_base + ".pdf", dpi=300, bbox_inches='tight')
plt.savefig(out_base + ".png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_base}.pdf / .png")
plt.show()
