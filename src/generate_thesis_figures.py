#!/usr/bin/env python3
"""
Generate two thesis figures:
1) Binary boundary map visualization with grid of 0/1 values at block corners
2) Block diagram of VVC-PPFF model input/output channels
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap


def generate_boundary_map_figure(output_path="boundary_map_viz.png"):
    """
    Visualize a binary boundary map zoomed into a corner between CU blocks.
    Shows a grid with '1' on boundaries and '0' in interiors.
    """
    h, w = 14, 18
    boundary = np.zeros((h, w), dtype=np.int32)
    
    boundary[7, :] = 1
    
    boundary[:8, 10] = 1
    
    boundary[8:, 5] = 1
    boundary[8:, 12] = 1
    
    boundary[0, :] = 1
    boundary[:, 0] = 1
    boundary[h-1, :] = 1
    boundary[:, w-1] = 1
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=150)
    
    cmap = ListedColormap(['#ffffff', '#cc3333'])
    ax.imshow(boundary, cmap=cmap, interpolation='nearest', aspect='equal')
    
    for i in range(h):
        for j in range(w):
            val = boundary[i, j]
            color = 'white' if val == 1 else 'black'
            fontweight = 'bold' if val == 1 else 'normal'
            ax.text(j, i, str(val), ha='center', va='center', 
                   fontsize=8, color=color, fontweight=fontweight)
    
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='#cccccc', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    ax.set_xticks(range(0, w, 2))
    ax.set_yticks(range(0, h, 2))
    ax.set_xlabel('Pozycja x [piksele]', fontsize=11)
    ax.set_ylabel('Pozycja y [piksele]', fontsize=11)
    ax.set_title('Binarna mapa granic bloków kodowych (fragment)', fontsize=12, fontweight='bold')
    
    ax.annotate('CU A', xy=(5, 3.5), fontsize=10, ha='center', va='center',
               color='#336699', fontweight='bold')
    ax.annotate('CU B', xy=(14, 3.5), fontsize=10, ha='center', va='center',
               color='#336699', fontweight='bold')
    ax.annotate('CU C', xy=(2.5, 11), fontsize=9, ha='center', va='center',
               color='#336699', fontweight='bold')
    ax.annotate('CU D', xy=(8.5, 11), fontsize=9, ha='center', va='center',
               color='#336699', fontweight='bold')
    ax.annotate('CU E', xy=(15, 11), fontsize=9, ha='center', va='center',
               color='#336699', fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor='#cc3333', edgecolor='black', label='1 (granica CU)'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='0 (wnętrze CU)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def generate_model_channels_figure(output_path="model_channels_diagram.png"):
    """
    Block diagram showing VVC-PPFF input channels (12ch) and output (3ch).
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=150)
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1.5, 8)
    ax.axis('off')
    
    input_groups = [
        ("Klatka bieżąca (Y, Cb, Cr)", 3, '#4488cc', ['Y', 'Cb', 'Cr']),
        ("Mapa QP", 1, '#66aa44', ['QP']),
        ("Mapy granic (luma, chroma)", 2, '#cc8844', ['bnd_L', 'bnd_C']),
        ("Klatka poprzednia (Y, Cb, Cr)", 3, '#9966cc', ['Y_prev', 'Cb_prev', 'Cr_prev']),
        ("Klatka następna (Y, Cb, Cr)", 3, '#cc6699', ['Y_next', 'Cb_next', 'Cr_next']),
    ]
    
    x_in = 0.0
    bar_w = 3.2
    bar_h = 0.38
    y_pos = 7.0
    
    ax.text(x_in + bar_w/2, 7.6, 'Wejście (12 kanałów)', fontsize=12, 
            fontweight='bold', ha='center')
    
    ch_idx = 0
    group_mids = []
    for grp_label, n_ch, color, ch_labels in input_groups:
        grp_start = y_pos
        for i in range(n_ch):
            ch_idx += 1
            rect = mpatches.FancyBboxPatch(
                (x_in, y_pos - bar_h/2), bar_w, bar_h,
                boxstyle="round,pad=0.03", facecolor=color, edgecolor='#333',
                alpha=0.75, linewidth=0.7
            )
            ax.add_patch(rect)
            ax.text(x_in + bar_w/2, y_pos, f'{ch_labels[i]}', fontsize=8, 
                   ha='center', va='center', color='white', fontweight='bold')
            y_pos -= bar_h + 0.04
        grp_mid = (grp_start + y_pos + bar_h + 0.04) / 2
        group_mids.append(grp_mid)
        ax.text(x_in - 0.15, grp_mid, grp_label, fontsize=8, 
               ha='right', va='center', color=color, fontweight='bold')
        y_pos -= 0.12
    
    mx, my, mw, mh = 5.2, 1.0, 3.8, 5.5
    model_rect = mpatches.FancyBboxPatch(
        (mx, my), mw, mh,
        boxstyle="round,pad=0.2", facecolor='#f5f5f5', edgecolor='#222',
        linewidth=2.5
    )
    ax.add_patch(model_rect)
    ax.text(mx + mw/2, my + mh - 0.6, 'VVC-PPFF', fontsize=15, 
           fontweight='bold', ha='center', va='center', color='#222')
    ax.text(mx + mw/2, my + mh/2, 
           '16 bloków FEB\n128 kanałów cech\nprogresywna fuzja\nsplot 3×3 + PReLU',
           fontsize=9, ha='center', va='center', color='#555', linespacing=1.4)
    ax.text(mx + mw/2, my + 0.45, '~2,78 M parametrów', fontsize=8, 
           ha='center', color='#999', style='italic')
    
    arr_y = 3.75
    ax.annotate('', xy=(mx - 0.05, arr_y), xytext=(x_in + bar_w + 0.1, arr_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax.text((x_in + bar_w + mx) / 2, arr_y + 0.3, 'concat\n→ Conv 1×1\n→ 128 ch',
           fontsize=8, ha='center', va='bottom', color='#666')
    
    ox = 11.0
    out_labels = ['Y poprawiony', 'Cb poprawiony', 'Cr poprawiony']
    out_y = 4.5
    
    ax.text(ox + bar_w/2, 7.6, 'Wyjście (3 kanały)', fontsize=12, 
            fontweight='bold', ha='center')
    
    for label in out_labels:
        rect = mpatches.FancyBboxPatch(
            (ox, out_y - bar_h/2), bar_w, bar_h,
            boxstyle="round,pad=0.03", facecolor='#2277aa', edgecolor='#333',
            alpha=0.8, linewidth=0.7
        )
        ax.add_patch(rect)
        ax.text(ox + bar_w/2, out_y, label, fontsize=8, 
               ha='center', va='center', color='white', fontweight='bold')
        out_y -= bar_h + 0.08
    
    ax.annotate('', xy=(ox - 0.05, arr_y), xytext=(mx + mw + 0.05, arr_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax.text((mx + mw + ox) / 2, arr_y + 0.3, 'Conv 1×1 → 3ch\n+ tanh\n+ residuum',
           fontsize=8, ha='center', va='bottom', color='#666')
    
    ax.annotate('', xy=(ox - 0.05, 1.8), xytext=(x_in + bar_w + 0.1, 1.8),
               arrowprops=dict(arrowstyle='->', color='#cc3333', lw=1.5, 
                              linestyle='dashed', connectionstyle='arc3,rad=-0.2'))
    ax.text(7.1, 0.8, 'globalne połączenie rezydualne (skip connection)\nwyjście = wejście_YUV + residuum', 
           fontsize=8, ha='center', color='#cc3333', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    generate_boundary_map_figure("boundary_map_viz.png")
    generate_model_channels_figure("model_channels_diagram.png")
    print("\nDone!")
