#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 make_combined_movie.py  (v3 — FEL 3D rotating surface)
==============================================================================
 يدمج صور البروتين (من PyMOL) مع لوحات FEL 3D / Variance / Cross-Correlation
 في فيديو MP4 واحد، ويولّد صور منفصلة لكل لوحة.

 التخطيط:
   ┌─────────────────────┬──────────────────┐
   │                     │  FEL 3D surface  │
   │   البروتين+الليكند   │  (يدور مع الفيديو│
   │   (صورة PyMOL        │  + نقطة متحركة)  │
   │    تلف 360°)         ├──────────────────┤
   │                     │ Variance │ Cross- │
   │                     │ (تراكمي) │ Corr.  │
   └─────────────────────┴──────────────────┘

 المخرجات:
   - final_movie.mp4     ← الفيديو الكامل
   - FEL_2D.png          ← صورة FEL ثنائية الأبعاد
   - FEL_3D.png          ← صورة FEL ثلاثية الأبعاد
   - variance.png        ← صورة PC contributions
   - crosscorr.png       ← صورة cross-correlation matrix

 GENERAL: يشتغل لأي بروتين+ligand.

 الاستخدام:
   python make_combined_movie.py --top MD.gro --traj MD_center.xtc \
          --frames-dir frames_protein --stride 10 \
          --sel "name CA" --out final_movie.mp4
==============================================================================
"""
import argparse, sys, os, glob
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
#  ثيم عام (أبيض — واضح للنشر)
# ─────────────────────────────────────────────────────────────────────────────
THEME = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "black", "font.size": 9,
}
TITLE_COLOR  = "#222222"
ACCENT_COLOR = "#c0392b"   # أحمر داكن للعناوين الفرعية
CUM_COLOR    = "#27ae60"   # أخضر للـ cumulative


# ─────────────────────────────────────────────────────────────────────────────
#  دوال التحليل
# ─────────────────────────────────────────────────────────────────────────────
def load_align(top, traj, sel, stride):
    print("📂  Loading trajectory ...")
    t = md.load(traj, top=top, stride=stride)
    print(f"      frames (stride={stride}) : {t.n_frames}")
    s = t.topology.select(sel)
    if len(s) == 0:
        sys.exit(f"❌  selection '{sel}' matched 0 atoms.")
    t = t.atom_slice(s)
    t.superpose(t, frame=0)
    return t


def run_pca(t):
    print("🔬  PCA ...")
    X  = t.xyz.reshape(t.n_frames, -1).astype(np.float64)
    Xc = X - X.mean(axis=0)
    pca  = PCA(n_components=min(10, t.n_frames, Xc.shape[1]), svd_solver="full")
    proj = pca.fit_transform(Xc)
    print(f"      PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    return pca, proj


def compute_fel(pc1, pc2, bins, temp):
    print("🌋  FEL ...")
    kT       = 0.0083145 * temp
    H, xe, ye = np.histogram2d(pc1, pc2, bins=bins)
    H  = gaussian_filter(H, sigma=1.0)
    P  = H / H.sum()
    Pm = np.where(P > 0, P, np.nan)
    G  = -kT * np.log(Pm / np.nanmax(Pm))
    G  = G - np.nanmin(G)
    G  = np.where(np.isnan(G), np.nanmax(G), G)
    return G.T, xe, ye


def compute_dccm(t):
    print("🔗  Cross-correlation ...")
    xyz  = t.xyz
    disp = xyz - xyz.mean(axis=0)
    dot  = np.einsum('fad,fbd->ab', disp, disp) / xyz.shape[0]
    d    = np.diag(dot)
    C    = dot / np.sqrt(np.outer(d, d))
    return C


# ─────────────────────────────────────────────────────────────────────────────
#  مشبكات الـ FEL للـ surface 3D
# ─────────────────────────────────────────────────────────────────────────────
def make_fel_meshgrid(G, xe, ye):
    xc = 0.5*(xe[:-1]+xe[1:])
    yc = 0.5*(ye[:-1]+ye[1:])
    XX, YY = np.meshgrid(xc, yc)
    return XX, YY, G


# ─────────────────────────────────────────────────────────────────────────────
#  صور ثابتة منفصلة
# ─────────────────────────────────────────────────────────────────────────────
def save_static_images(G, xe, ye, C, pca, proj, out_dir):
    print("🖼️   Saving static images ...")
    pc1, pc2 = proj[:,0], proj[:,1]
    var1 = pca.explained_variance_ratio_[0]*100
    var2 = pca.explained_variance_ratio_[1]*100
    XX, YY, ZZ = make_fel_meshgrid(G, xe, ye)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]

    # ── FEL 2D ───────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6,5), dpi=150)
    plt.rcParams.update(THEME)
    im = ax2.imshow(G, origin="lower", extent=extent, aspect="auto", cmap="inferno")
    ax2.contour(G, levels=8, extent=extent, cmap="coolwarm", linewidths=0.8)
    ax2.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=11)
    ax2.set_title("Free Energy Landscape (2D)", fontsize=13, fontweight="bold")
    cb = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("ΔG (kJ/mol)", fontsize=10)
    ax2.scatter(pc1, pc2, c="white", s=1, alpha=0.15)
    path2d = os.path.join(out_dir, "FEL_2D.png")
    fig2.savefig(path2d, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"      ✅ {path2d}")

    # ── FEL 3D ───────────────────────────────────────────────────────────────
    fig3 = plt.figure(figsize=(7, 6), dpi=150)
    ax3  = fig3.add_subplot(111, projection="3d")
    norm = Normalize(vmin=ZZ.min(), vmax=ZZ.max())
    surf = ax3.plot_surface(XX, YY, ZZ, facecolors=cm.inferno(norm(ZZ)),
                            rstride=1, cstride=1, linewidth=0,
                            antialiased=True, alpha=0.95)
    ax3.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=9)
    ax3.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=9)
    ax3.set_zlabel("ΔG (kJ/mol)", fontsize=9)
    ax3.set_title("Free Energy Landscape (3D)", fontsize=13, fontweight="bold", pad=12)
    ax3.view_init(elev=28, azim=-60)
    ax3.set_facecolor("white")
    fig3.patch.set_facecolor("white")
    sm = cm.ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([])
    fig3.colorbar(sm, ax=ax3, shrink=0.5, pad=0.08, label="ΔG (kJ/mol)")
    path3d = os.path.join(out_dir, "FEL_3D.png")
    fig3.savefig(path3d, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"      ✅ {path3d}")

    # ── Variance ──────────────────────────────────────────────────────────────
    ns  = min(8, len(pca.explained_variance_ratio_))
    evr = pca.explained_variance_ratio_[:ns]*100
    cum = np.cumsum(evr); bx = np.arange(1, ns+1)
    figv, axv = plt.subplots(figsize=(5,4), dpi=150)
    plt.rcParams.update(THEME)
    axv.bar(bx, evr, color="#7f8c8d", edgecolor="black")
    axv.set_xlabel("PC", fontsize=11); axv.set_ylabel("Variance (%)", fontsize=11)
    axv.set_title("PC Contributions", fontsize=13, fontweight="bold")
    av2 = axv.twinx()
    av2.plot(bx, cum, "-o", color=CUM_COLOR, ms=5, lw=2)
    av2.set_ylabel("Cumulative (%)", color=CUM_COLOR, fontsize=11)
    av2.tick_params(axis="y", colors=CUM_COLOR)
    av2.set_ylim(0,105)
    pathv = os.path.join(out_dir, "variance.png")
    figv.savefig(pathv, dpi=150, bbox_inches="tight")
    plt.close(figv)
    print(f"      ✅ {pathv}")

    # ── Cross-Correlation ────────────────────────────────────────────────────
    figd, axd = plt.subplots(figsize=(5,5), dpi=150)
    plt.rcParams.update(THEME)
    imd = axd.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    axd.set_xlabel("Residue", fontsize=11); axd.set_ylabel("Residue", fontsize=11)
    axd.set_title("Residue Cross-Correlation", fontsize=13, fontweight="bold")
    figd.colorbar(imd, ax=axd, fraction=0.046, pad=0.04)
    pathd = os.path.join(out_dir, "crosscorr.png")
    figd.savefig(pathd, dpi=150, bbox_inches="tight")
    plt.close(figd)
    print(f"      ✅ {pathd}")


# ─────────────────────────────────────────────────────────────────────────────
#  بناء الفيديو (FEL 3D يدور)
# ─────────────────────────────────────────────────────────────────────────────
def build_movie(prot_imgs, pca, proj, G, xe, ye, C, n, args):
    print("🖌️   Building movie with 3D rotating FEL ...")
    plt.rcParams.update(THEME)

    pc1, pc2 = proj[:,0], proj[:,1]
    var1 = pca.explained_variance_ratio_[0]*100
    var2 = pca.explained_variance_ratio_[1]*100
    XX, YY, ZZ = make_fel_meshgrid(G, xe, ye)
    norm_fel = Normalize(vmin=ZZ.min(), vmax=ZZ.max())

    ns  = min(8, len(pca.explained_variance_ratio_))
    evr = pca.explained_variance_ratio_[:ns]*100
    cum = np.cumsum(evr); bx = np.arange(1, ns+1)

    fig = plt.figure(figsize=(13, 7.3), dpi=100)
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.30,
                  left=0.03, right=0.96, top=0.90, bottom=0.08,
                  width_ratios=[1.35, 1, 1])
    fig.suptitle("Protein–Ligand Dynamics:  Structure → PCA → Free Energy",
                 color=TITLE_COLOR, fontsize=15, fontweight="bold", y=0.965)

    # ── لوحة البروتين (يسار) ─────────────────────────────────────────────────
    ax_prot = fig.add_subplot(gs[:, 0])
    ax_prot.axis("off")
    ax_prot.set_title("Structure Dynamics", color=TITLE_COLOR,
                      fontsize=10, fontweight="bold")
    im0          = mpimg.imread(prot_imgs[0])
    prot_artist  = ax_prot.imshow(im0)

    # ── FEL 3D (يمين-فوق، يدور) ──────────────────────────────────────────────
    ax_fel = fig.add_subplot(gs[0, 1:], projection="3d")
    ax_fel.set_facecolor("white")
    surf = ax_fel.plot_surface(XX, YY, ZZ,
                               facecolors=cm.inferno(norm_fel(ZZ)),
                               rstride=1, cstride=1,
                               linewidth=0, antialiased=True, alpha=0.92)
    # نقطة متحركة على السطح
    pt3d, = ax_fel.plot([], [], [], "o", color="yellow",
                        markersize=8, markeredgecolor="black", zorder=10)
    trail3d, = ax_fel.plot([], [], [], "-", color="cyan",
                           lw=1.2, alpha=0.8, zorder=9)
    ax_fel.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=8, labelpad=2)
    ax_fel.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=8, labelpad=2)
    ax_fel.set_zlabel("ΔG (kJ/mol)", fontsize=8, labelpad=2)
    ax_fel.set_title("Free Energy Landscape (3D)",
                     color=TITLE_COLOR, fontsize=10, fontweight="bold", pad=8)
    ax_fel.view_init(elev=28, azim=-60)
    sm = cm.ScalarMappable(cmap="inferno", norm=norm_fel)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_fel, shrink=0.45, pad=0.08,
                 label="ΔG (kJ/mol)", aspect=15)

    # ارتفاع النقطة على السطح (interpolation من ZZ)
    def get_z(xi, yi):
        xc = 0.5*(xe[:-1]+xe[1:])
        yc = 0.5*(ye[:-1]+ye[1:])
        ix = np.argmin(np.abs(xc - xi))
        iy = np.argmin(np.abs(yc - yi))
        return ZZ[iy, ix]

    # ── Variance (يمين-تحت-يسار) ──────────────────────────────────────────────
    ax_var = fig.add_subplot(gs[1, 1])
    ax_var.set_facecolor("white")
    ax_var.bar(bx, evr, color="#95a5a6", edgecolor="black")
    ax_var.set_xlabel("PC", fontsize=9)
    ax_var.set_ylabel("Variance (%)", fontsize=9)
    ax_var.set_title("PC Contributions", color=TITLE_COLOR,
                     fontsize=10, fontweight="bold")
    ax_var.set_xlim(0.4, ns+0.6)
    ax_v2 = ax_var.twinx()
    ax_v2.set_ylim(0, 105); ax_v2.set_xlim(0.4, ns+0.6)
    ax_v2.set_ylabel("Cumulative (%)", color=CUM_COLOR, fontsize=9)
    ax_v2.tick_params(axis="y", colors=CUM_COLOR)
    cum_line, = ax_v2.plot([], [], "-o", color=CUM_COLOR, ms=4, lw=2)

    # ── Cross-Correlation (يمين-تحت-يمين) ────────────────────────────────────
    ax_dcc = fig.add_subplot(gs[1, 2])
    ax_dcc.set_facecolor("white")
    imd = ax_dcc.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax_dcc.set_xlabel("Residue", fontsize=9)
    ax_dcc.set_ylabel("Residue", fontsize=9)
    ax_dcc.set_title("Cross-Correlation", color=TITLE_COLOR,
                     fontsize=10, fontweight="bold")
    fig.colorbar(imd, ax=ax_dcc, fraction=0.046, pad=0.02)

    # ── دالة التحديث ──────────────────────────────────────────────────────────
    total_azim = 360.0
    start_azim = -60.0

    def update(frame):
        # البروتين
        img = mpimg.imread(prot_imgs[frame])
        prot_artist.set_data(img)

        # دوران الـ FEL 3D
        azim = start_azim + (frame / n) * total_azim
        ax_fel.view_init(elev=28, azim=azim)

        # النقطة على السطح 3D
        xi, yi = pc1[frame], pc2[frame]
        zi = get_z(xi, yi)
        pt3d.set_data([xi], [yi]); pt3d.set_3d_properties([zi + 0.05])
        lo = max(0, frame-40)
        zs = [get_z(pc1[f], pc2[f]) for f in range(lo, frame+1)]
        trail3d.set_data(pc1[lo:frame+1], pc2[lo:frame+1])
        trail3d.set_3d_properties(zs)

        # cumulative تدريجي
        k = max(1, int(round((frame+1)/n * ns)))
        cum_line.set_data(bx[:k], cum[:k])

        return prot_artist, pt3d, trail3d, cum_line

    print(f"🎞️   rendering {n} frames @ {args.fps} fps ...")
    anim = animation.FuncAnimation(fig, update, frames=n,
                                   interval=1000/args.fps, blit=False)
    try:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=3500, codec="libx264")
        anim.save(args.out, writer=writer, dpi=100)
        print(f"\n✅  Movie saved → {args.out}")
    except Exception as e:
        gif = args.out.rsplit(".", 1)[0] + ".gif"
        print(f"⚠️  ffmpeg failed ({e}); GIF → {gif}")
        anim.save(gif, writer="pillow", fps=args.fps)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",        required=True)
    ap.add_argument("--traj",       required=True)
    ap.add_argument("--frames-dir", default="frames_protein")
    ap.add_argument("--stride",     type=int,   default=10)
    ap.add_argument("--sel",        default="name CA")
    ap.add_argument("--out",        default="final_movie.mp4")
    ap.add_argument("--bins",       type=int,   default=80)
    ap.add_argument("--temp",       type=float, default=300.0)
    ap.add_argument("--fps",        type=int,   default=15)
    args = ap.parse_args()

    # مجلد المخرجات (نفس مجلد الـ mp4)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."

    # صور البروتين
    prot_imgs = sorted(glob.glob(os.path.join(args.frames_dir, "prot_*.png")))
    if not prot_imgs:
        sys.exit(f"❌  no frames in {args.frames_dir}/")
    print(f"🖼️   found {len(prot_imgs)} protein frames.")

    # التحليل
    t        = load_align(args.top, args.traj, args.sel, args.stride)
    pca, proj = run_pca(t)
    pc1, pc2  = proj[:,0], proj[:,1]
    G, xe, ye = compute_fel(pc1, pc2, args.bins, args.temp)
    C         = compute_dccm(t)

    n = min(len(prot_imgs), len(proj))
    print(f"🎞️   syncing {n} frames.")

    # الصور الثابتة المنفصلة
    save_static_images(G, xe, ye, C, pca, proj, out_dir)

    # الفيديو
    build_movie(prot_imgs, pca, proj, G, xe, ye, C, n, args)

    print("\n🏁  All done.")
    print(f"   📹 Movie  : {args.out}")
    print(f"   🖼️  FEL 2D : {os.path.join(out_dir, 'FEL_2D.png')}")
    print(f"   🖼️  FEL 3D : {os.path.join(out_dir, 'FEL_3D.png')}")
    print(f"   🖼️  Variance: {os.path.join(out_dir, 'variance.png')}")
    print(f"   🖼️  CrossCorr: {os.path.join(out_dir, 'crosscorr.png')}")


if __name__ == "__main__":
    main()
