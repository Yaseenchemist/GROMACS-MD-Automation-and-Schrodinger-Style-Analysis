import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import AllChem
import os, sys

# ============================================================
XVG_FILE  = "rmsfl_heavy.xvg"
MOL2_FILE = "LIG.mol2"
OUTPUT_PNG = "ligand_rmsf.png"
OUTPUT_SVG = "ligand_rmsf.svg"
DPI        = 300
SCALE_MOL  = 1.6
# ============================================================

ELEMENT_COLORS = {
    "N":  "#0000FF",
    "O":  "#FF0000",
    "S":  "#B8860B",
    "F":  "#00BB00",
    "Cl": "#00BB00",
    "Br": "#8B2500",
    "I":  "#660099",
    "P":  "#FF8000",
    "C":  "#000000",
}

def read_xvg(filename):
    x_data, y_data = [], []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x_data.append(float(parts[0]))
                    y_data.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x_data), np.array(y_data)


def load_mol2(filename):
    mol = Chem.MolFromMol2File(filename, removeHs=False, sanitize=False)
    if mol is None:
        sys.exit(f"[خطأ] تعذّر قراءة {filename}")
    Chem.SanitizeMol(mol, catchErrors=True)
    mol = Chem.RemoveHs(mol, sanitize=False)
    print(f"   ✓ mol2 | ذرات Heavy: {mol.GetNumAtoms()}")
    print(f"   أروماتية: {sum(1 for b in mol.GetBonds() if b.GetIsAromatic())} | "
          f"مزدوجة: {sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble()==2.0)}")

    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x * SCALE_MOL,
                                 pos.y * SCALE_MOL,
                                 pos.z))
    coords = {}
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = (pos.x, pos.y)

    return mol, coords


def shorten(xi, yi, xj, yj, s):
    dx, dy = xj-xi, yj-yi
    L = np.sqrt(dx**2+dy**2)
    if L == 0: return xi, yi, xj, yj
    ux, uy = dx/L, dy/L
    return xi+ux*s, yi+uy*s, xj-ux*s, yj-uy*s

def perp_toward_center(x1, y1, x2, y2, cx, cy, d):
    mx, my = (x1+x2)/2, (y1+y2)/2
    dcx, dcy = cx-mx, cy-my
    L = np.sqrt(dcx**2+dcy**2)
    if L == 0: return 0, 0
    return (dcx/L)*d, (dcy/L)*d

def perp_offset(x1, y1, x2, y2, d):
    dx, dy = x2-x1, y2-y1
    L = np.sqrt(dx**2+dy**2)
    if L == 0: return 0, 0
    return (-dy/L)*d, (dx/L)*d


def draw_molecule_on_ax(ax, mol, coords):
    ring_info  = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    bond_rings = ring_info.BondRings()

    ring_centers = []
    for ring in atom_rings:
        ring_centers.append((
            np.mean([coords[i][0] for i in ring]),
            np.mean([coords[i][1] for i in ring])
        ))

    bond_to_ring = {}
    for ri, (rbonds, ratoms) in enumerate(zip(bond_rings, atom_rings)):
        arom_n = sum(1 for b in rbonds
                     if mol.GetBondWithIdx(b).GetIsAromatic())
        if arom_n >= len(rbonds) // 2:
            for b in rbonds:
                if mol.GetBondWithIdx(b).GetIsAromatic():
                    bond_to_ring[b] = ri

    aromatic_inner = set()
    for ri, rbonds in enumerate(bond_rings):
        arom_bonds = [b for b in rbonds if b in bond_to_ring]
        for k, b in enumerate(arom_bonds):
            if k % 2 == 0:
                aromatic_inner.add(b)

    C  = "#222222"
    LW = 2.0
    S  = 0.30
    IS = 0.22
    D  = 0.16

    for bond in mol.GetBonds():
        bidx    = bond.GetIdx()
        i, j    = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        xi, yi  = coords[i]
        xj, yj  = coords[j]
        is_arom = bond.GetIsAromatic()
        btype   = bond.GetBondTypeAsDouble()

        xs, ys, xe, ye = shorten(xi, yi, xj, yj, S)
        dx = xj-xi; dy = yj-yi
        L  = np.sqrt(dx**2+dy**2)
        ux, uy = (dx/L, dy/L) if L > 0 else (0, 0)

        if is_arom and bidx in bond_to_ring:
            ri = bond_to_ring[bidx]
            cx, cy = ring_centers[ri]
            ax.plot([xs, xe], [ys, ye],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)
            if bidx in aromatic_inner:
                ox, oy = perp_toward_center(xs, ys, xe, ye, cx, cy, D)
                ax.plot([xs+ox+ux*IS, xe+ox-ux*IS],
                        [ys+oy+uy*IS, ye+oy-uy*IS],
                        color=C, lw=LW, solid_capstyle='round', zorder=1)

        elif btype == 2.0:
            ox, oy = perp_offset(xs, ys, xe, ye, D)
            ax.plot([xs+ox, xe+ox], [ys+oy, ye+oy],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)
            ax.plot([xs-ox, xe-ox], [ys-oy, ye-oy],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)

        elif btype == 3.0:
            ox, oy = perp_offset(xs, ys, xe, ye, D*1.2)
            ax.plot([xs,    xe   ], [ys,    ye   ],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)
            ax.plot([xs+ox, xe+ox], [ys+oy, ye+oy],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)
            ax.plot([xs-ox, xe-ox], [ys-oy, ye-oy],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)

        else:
            ax.plot([xs, xe], [ys, ye],
                    color=C, lw=LW, solid_capstyle='round', zorder=1)

    all_x = [v[0] for v in coords.values()]
    all_y = [v[1] for v in coords.values()]
    scale = max(max(all_x)-min(all_x), max(all_y)-min(all_y))
    fs    = max(8.5, min(13.0, 110/scale))

    for atom in mol.GetAtoms():
        idx   = atom.GetIdx()
        color = ELEMENT_COLORS.get(atom.GetSymbol(), "#000000")
        x, y  = coords[idx]
        ax.text(x, y, str(idx+1),
                color=color, fontsize=fs, fontweight="bold",
                ha="center", va="center", zorder=3)

    margin = 1.0
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_y)-margin, max(all_y)+margin)
    ax.set_aspect("equal")
    ax.axis("off")


def build_figure(xvg_file, mol2_file):
    """بناء الشكل وإرجاعه بدون حفظ — يُستخدم مرتين للـ PNG و SVG"""

    print(f"[1/3] قراءة RMSF: {xvg_file}")
    _, rmsf_raw = read_xvg(xvg_file)
    rmsf_A    = rmsf_raw * 10.0 if rmsf_raw.max() < 5.0 else rmsf_raw
    n_atoms   = len(rmsf_A)
    atom_nums = np.arange(1, n_atoms+1)
    print(f"   {n_atoms} نقطة | min={rmsf_A.min():.2f} Å  max={rmsf_A.max():.2f} Å")

    print(f"\n[2/3] تحميل الجزيء: {mol2_file}")
    mol, coords = load_mol2(mol2_file)
    n_heavy = mol.GetNumAtoms()
    if n_heavy != n_atoms:
        print(f"   [تحذير] {n_heavy} ذرة ≠ {n_atoms} نقطة RMSF")

    print("\n[3/3] بناء الشكل...")
    MAROON = "#8B1A4A"

    fig = plt.figure(figsize=(14, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.6, 1.0],
                            hspace=0.0, top=0.96, bottom=0.07,
                            left=0.08, right=0.97)

    ax_mol = fig.add_subplot(gs[0])
    ax_mol.set_title("Ligand RMSF", fontsize=19, fontweight="bold", pad=10)
    draw_molecule_on_ax(ax_mol, mol, coords)

    ax = fig.add_subplot(gs[1])
    ax.plot(atom_nums, rmsf_A, color=MAROON, linewidth=2.4,
            solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(0.5, n_atoms+0.5)
    y_max = np.ceil(rmsf_A.max()) + 0.5
    ax.set_ylim(0, y_max)
    ax.set_xticks(atom_nums)
    ax.set_xticklabels([str(i) for i in atom_nums], fontsize=9.0)
    ax.set_yticks(np.arange(0, int(y_max)+1, 1))
    ax.tick_params(axis="both", labelsize=11, length=4)
    ax.set_ylabel("RMSF (Å)", fontsize=13, labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[Patch(facecolor=MAROON, edgecolor=MAROON,
                             label="Fit Ligand on Protein")],
              loc="lower right", fontsize=11, frameon=True,
              framealpha=0.9, edgecolor="#aaaaaa",
              handlelength=1.8, handleheight=1.0)

    return fig


def create_rmsf_figure(xvg_file, mol2_file,
                        output_png, output_svg, dpi=300):

    fig = build_figure(xvg_file, mol2_file)

    # ── حفظ PNG ─────────────────────────────────────────────
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none",
                format="png")
    print(f"✅ PNG محفوظ: {output_png}  (DPI={dpi})")

    # ── حفظ SVG ─────────────────────────────────────────────
    fig.savefig(output_svg, bbox_inches="tight",
                facecolor="white", edgecolor="none",
                format="svg",
                metadata={"Creator": "ligand_rmsf_plot"})
    print(f"✅ SVG محفوظ: {output_svg}  (وضوح لا محدود)")

    plt.close(fig)
    print("\n🎉 تم الحفظ بكلا الصيغتين بنجاح!")


if __name__ == "__main__":
    create_rmsf_figure(XVG_FILE, MOL2_FILE,
                       OUTPUT_PNG, OUTPUT_SVG, DPI)
