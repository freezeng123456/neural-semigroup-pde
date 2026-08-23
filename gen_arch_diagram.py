"""Generate a clean network architecture diagram for the Beamer presentation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

DPI = 250
FIG_W, FIG_H = 16, 8.5
BG = '#FFFFFF'

def box(ax, cx, cy, w, h, text, fc, ec='#555555', lw=1.2, fs=11,
        tc='#1a1a1a', zo=3, bold=False):
    x, y = cx - w/2, cy - h/2
    fp = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zo)
    ax.add_patch(fp)
    wt = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, zorder=zo+1, weight=wt)

def arr(ax, x1, y1, x2, y2, c='#333333', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=2)

def lbl(ax, x, y, t, fs=9, c='#666666'):
    ax.text(x, y, t, ha='center', va='center', fontsize=fs, color=c, zorder=5)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-8.0, 1.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG)

# ── Row 1: Input / Output ──
Y0, H0 = 0.0, 0.55
box(ax, 1.5, Y0, 2.8, H0, r'$u \in (0,1)^N$', '#D6EAF8', fs=12)
box(ax, 5.5, Y0, 3.0, H0, r'$z_0 = g^{-1}(u)$', '#D5F5E3', fs=12)
box(ax, 11.5, Y0, 1.8, H0, r'$z_T$', '#FDEBD0', fs=12, bold=True)
box(ax, 14.8, Y0, 2.8, H0, r'$\hat{u} = g(z_T)$', '#D6EAF8', fs=12)
arr(ax, 2.9, Y0, 4.0, Y0); lbl(ax, 3.45, Y0+0.32, 'logit', fs=8.5)
arr(ax, 12.4, Y0, 13.4, Y0); lbl(ax, 12.9, Y0+0.32, 'sigmoid', fs=8.5)

# ── Row 2: Three paths ──
YT, YB = -1.8, -3.6
HP = 0.6

# V_theta (x=1.5)
lbl(ax, 1.5, YT+0.5, 'scalar MLP', fs=8.5, c='#8E44AD')
box(ax, 1.5, YT, 2.8, HP, r'$V_\theta(z_i)$', '#E8DAEF', fs=11)
arr(ax, 1.5, YT-HP/2, 1.5, YB+HP/2, c='#8E44AD', lw=1.2)
box(ax, 1.5, YB, 2.4, HP, r"$V'_\theta(z_i)$", '#E8DAEF', fs=11)

# Interaction (x=5.5)
lbl(ax, 5.5, YT+0.5, 'learned embeddings', fs=8.5, c='#148F77')
box(ax, 5.5, YT, 3.8, HP, r'$a_{ij} = m_{ij}\,\mathrm{softplus}(e_i^T e_j)$',
    '#D1F2EB', fs=10)
arr(ax, 5.5, YT-HP/2, 5.5, YB+HP/2, c='#148F77', lw=1.2)
box(ax, 5.5, YB, 3.8, HP, r'$2\,\sum_j a_{ij}(z_i - z_j)$', '#D1F2EB', fs=10)

# K_theta (x=11.5)
lbl(ax, 11.5, YT+0.5, 'StencilMLP + softplus', fs=8.5, c='#7E5109')
box(ax, 11.5, YT, 3.2, HP, r'$k_{\theta,i}(z)$', '#F9E79F', fs=11)
arr(ax, 11.5, YT-HP/2, 11.5, YB+HP/2, c='#7E5109', lw=1.2)
box(ax, 11.5, YB, 2.8, HP, r'$K_\theta = \mathrm{diag}(k_i)$', '#F9E79F', fs=11)

# z0 splits
arr(ax, 5.5, Y0-H0/2, 5.5, YT+HP/2, c='#999', lw=1.2)
arr(ax, 5.5, YT+HP/2, 1.5, YT+HP/2, c='#999', lw=1.2)
arr(ax, 5.5, YT+HP/2, 11.5, YT+HP/2, c='#999', lw=1.2)

# ── Row 3: Combine ──
Y3 = -5.5; H3 = 0.6

# + operator
ax.plot(3.5, Y3, 'o', ms=18, color='#F5B7B1', mec='#C0392B', mew=1.5, zorder=4)
ax.plot([3.35, 3.65], [Y3, Y3], color='#C0392B', lw=2, zorder=5)
ax.plot([3.5, 3.5], [Y3-0.15, Y3+0.15], color='#C0392B', lw=2, zorder=5)

# nabla Psi
box(ax, 6.0, Y3, 2.8, H3, r'$[\nabla\Psi_\theta]_i$', '#FADBD8', fs=11, ec='#C0392B')

# x operator
ax.plot(8.8, Y3, 'o', ms=16, color='#F5B7B1', mec='#C0392B', mew=1.5, zorder=4)
ax.plot([8.63, 8.97], [Y3-0.12, Y3+0.12], color='#C0392B', lw=2, zorder=5)
ax.plot([8.63, 8.97], [Y3+0.12, Y3-0.12], color='#C0392B', lw=2, zorder=5)

# - operator
ax.plot(10.3, Y3, 'o', ms=14, color='#F5B7B1', mec='#C0392B', mew=1.5, zorder=4)
ax.plot([10.15, 10.45], [Y3, Y3], color='#C0392B', lw=2.5, zorder=5)

# dz/ds
box(ax, 12.0, Y3, 2.4, H3, r'$dz/ds$', '#FDEBD0', fs=12)

# RK4
box(ax, 14.8, Y3, 2.8, H3*1.1, 'RK4 integrator', '#FDEBD0', fs=11, bold=True, ec='#7E5109', lw=1.8)
lbl(ax, 14.8, Y3-0.5, '30 steps: z0 -> zT', fs=8.5, c='#7E5109')

# Row 3 arrows
arr(ax, 1.5, YB-HP/2, 1.5, Y3+0.3, c='#8E44AD', lw=1.2)
arr(ax, 1.5, Y3+0.3, 3.2, Y3, c='#8E44AD', lw=1.2)
arr(ax, 5.5, YB-HP/2, 5.5, Y3+0.3, c='#148F77', lw=1.2)
arr(ax, 5.5, Y3+0.3, 3.8, Y3, c='#148F77', lw=1.2)
arr(ax, 3.8, Y3, 4.6, Y3, c='#C0392B')
arr(ax, 7.4, Y3, 8.4, Y3, c='#C0392B')
arr(ax, 9.2, Y3, 10.0, Y3, c='#C0392B')
arr(ax, 10.6, Y3, 10.8, Y3, c='#C0392B')
arr(ax, 13.2, Y3, 13.4, Y3)
arr(ax, 11.5, YB-HP/2, 11.5, Y3+0.3, c='#7E5109', lw=1.2)
arr(ax, 11.5, Y3+0.3, 9.2, Y3, c='#7E5109', lw=1.2)

# RK4 -> zT
arr(ax, 15.8, Y3+0.2, 15.8, Y0-H0/2, c='#7E5109', lw=1.2)
arr(ax, 15.8, Y0-H0/2, 12.4, Y0-H0/2, c='#7E5109', lw=1.2)

# ── Guarantee annotations ──
for gy, gt in [(-1.0, r'$g=\sigma$ => admissibility'),
               (-2.5, r'$K_\theta \succ 0$ => dissipation'),
               (-4.0, 'ODE flow => semigroup')]:
    box(ax, 15.5, gy, 3.2, 0.45, gt, '#EAFAF1', ec='#27AE60', lw=1.0, fs=9, tc='#1E8449')

out = '/home/shuixinf/projects/DC-PINNs_LaTeX_arxiv_2604.13723/architecture_diagram.png'
fig.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.15, facecolor=BG)
plt.close()
print(f"Saved: {out}")
