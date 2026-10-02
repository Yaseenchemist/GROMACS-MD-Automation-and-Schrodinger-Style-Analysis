#!/usr/bin/env python3
"""
PCA_FEL_ANALYZER.py
===================
Flexible GROMACS 2025 PCA / FEL workflow for one MD system or a comparison
between two MD systems.

Designed from the workflow supplied in tutorial.zip (pca_gromacs.py,
PCA_gibbs_energy.txt, PCA/FEL plotting notebooks), but removes hard-coded
filenames, group numbers, paths, and C-alpha assumptions.

Core capabilities
-----------------
SINGLE SYSTEM
  * User-defined analysis region from an .ndx group OR MDAnalysis selection.
  * Optional atom-mode restriction: all / heavy / backbone / calpha.
  * Selected trajectory + first-frame reference generation.
  * Covariance matrix, eigenvalues, eigenvectors, average structure.
  * Scree plot and cumulative variance.
  * Projection on first N PCs and cosine content.
  * Time-convergence PCA/cosine-content analysis.
  * PC1-PC2, PC1-PC3, PC2-PC3 projections.
  * 3D PC1-PC2-PC3 projection.
  * Extreme structures along the first PCs (for porcupine/mode-vector plots).
  * Free-energy landscapes (FEL) for PC1-PC2, PC1-PC3, PC2-PC3 using gmx sham.
  * Probability, enthalpy, entropy XPM outputs from gmx sham.
  * 2D FEL contour and 3D FEL surface plots where XPM parsing succeeds.
  * Covariance heatmap where XPM parsing succeeds.
  * Human-readable summary and machine-readable manifest.

TWO-SYSTEM COMPARISON
  * Runs the complete single-system workflow for both systems.
  * Requires the selected atom sets to be compatible (same atom count and,
    by default, same resid/resname/atom-name signature).
  * Builds a common PCA basis from the concatenated selected trajectories.
  * Projects each system onto the SAME eigenvectors.
  * Overlay plots in common PC1-PC2, PC1-PC3 and PC2-PC3 spaces.
  * FEL for each system in the SAME PCA basis.
  * Quantitative conformational-space overlap (2D histogram overlap).
  * Centroid separation in common PCA space.
  * Eigenvector inner-product matrix and RMSIP estimate for the first N PCs.
  * Comparison summary and CSV metrics.

Important interpretation note
-----------------------------
For a scientifically meaningful comparison, the two systems must use the SAME
structural degrees of freedom. For example, comparing Apo vs Holo should use
matching protein atoms / matching binding-pocket atoms in both systems. Do not
include a ligand in one PCA selection if the second system has no corresponding
ligand atoms.

Requirements
------------
  * GROMACS 2025.x available as `gmx` (or use --gmx /path/to/gmx)
  * Python environment with: MDAnalysis, numpy, matplotlib

Example: complex only
---------------------
python3 PCA_FEL_ANALYZER.py single \
  --name Complex \
  --tpr MD.tpr \
  --traj MD_center.xtc \
  --ndx combined_full.ndx \
  --group BindingSite \
  --atoms calpha \
  --temperature 300 \
  --output Complex_PCA_FEL

Example: Apo vs Complex
-----------------------
python3 PCA_FEL_ANALYZER.py compare \
  --name1 Apo \
  --tpr1 APO.tpr --traj1 APO_center.xtc --ndx1 APO.ndx \
  --group1 BindingSite --atoms1 calpha \
  --name2 Complex \
  --tpr2 MD.tpr --traj2 MD_center.xtc --ndx2 combined_full.ndx \
  --group2 BindingSite --atoms2 calpha \
  --temperature 300 \
  --output Comparison_Apo_vs_Complex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: MDAnalysis is required. Activate your mda2025 environment first.\n"
        f"Import error: {exc}"
    )


SCRIPT_VERSION = "2026-09-16-flex-pca-fel-v1.4.4-exact-index-identity-matching"
PC_PAIRS = ((1, 2), (1, 3), (2, 3))


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class SystemSpec:
    name: str
    tpr: Path
    traj: Path
    ndx: Optional[Path]
    group: Optional[str]
    selection: Optional[str]
    atom_mode: str
    resids: Optional[List[int]]
    start_ns: Optional[float]
    stop_ns: Optional[float]
    frame_stride: int


@dataclass
class PreparedSystem:
    spec: SystemSpec
    outdir: Path
    selected_xtc: Path
    ref_pdb: Path
    selection_ndx: Path
    atom_table: Path
    n_atoms: int
    n_frames: int
    first_time_ns: float
    last_time_ns: float
    signature: List[Tuple[int, str, str]]


# =============================================================================
# Generic helpers
# =============================================================================

def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s.strip("_") or "system"


def ensure_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def run_cmd(
    cmd: Sequence[str],
    *,
    stdin: Optional[str] = None,
    cwd: Optional[Path] = None,
    label: str = "",
    log_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    if label:
        print(f"[run] {label}")
    print("      " + " ".join(map(str, cmd)))
    p = subprocess.run(
        list(map(str, cmd)),
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(p.stdout or "", encoding="utf-8")
    if p.stdout:
        lines = p.stdout.rstrip().splitlines()
        for line in lines[-12:]:
            print("      " + line)
    if check and p.returncode != 0:
        tail = "\n".join((p.stdout or "").splitlines()[-60:])
        raise RuntimeError(
            f"Command failed ({p.returncode}) during: {label or cmd[0]}\n"
            f"Command: {' '.join(map(str, cmd))}\n\n{tail}"
        )
    return p


def find_gmx(gmx_arg: str) -> str:
    if "/" in gmx_arg:
        p = Path(gmx_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GROMACS executable not found: {p}")
        return str(p)
    hit = shutil.which(gmx_arg)
    if not hit:
        raise FileNotFoundError(
            f"Could not find '{gmx_arg}' in PATH. Activate/source GROMACS 2025 first, "
            "or provide --gmx /full/path/to/gmx."
        )
    return hit


# =============================================================================
# Index handling and flexible selection
# =============================================================================

def parse_ndx(path: Path) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    current: Optional[str] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            groups[current] = []
            continue
        if current is not None:
            for tok in line.split():
                try:
                    groups[current].append(int(tok))
                except ValueError:
                    pass
    return groups


def resolve_ndx_group(groups: Dict[str, List[int]], requested: str) -> Tuple[str, List[int]]:
    if requested in groups:
        return requested, groups[requested]
    lower = {k.lower(): k for k in groups}
    key = lower.get(requested.lower())
    if key is not None:
        return key, groups[key]
    available = "\n  ".join(groups.keys())
    raise ValueError(
        f"Index group '{requested}' not found. Available groups:\n  {available}"
    )


def write_ndx(path: Path, name: str, atom_indices_1based: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"[ {name} ]\n")
        vals = list(map(int, atom_indices_1based))
        for i in range(0, len(vals), 15):
            f.write(" ".join(map(str, vals[i:i + 15])) + "\n")


def _atom_mode_mask(ag, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "all":
        return np.ones(ag.n_atoms, dtype=bool)
    if mode == "calpha":
        return np.asarray([a.name.upper() == "CA" for a in ag.atoms], dtype=bool)
    if mode == "backbone":
        allowed = {"N", "CA", "C", "O", "OXT"}
        return np.asarray([a.name.upper() in allowed for a in ag.atoms], dtype=bool)
    if mode == "heavy":
        mask = []
        for a in ag.atoms:
            keep = True
            try:
                keep = float(a.mass) > 1.5
            except Exception:
                nm = a.name.upper()
                keep = not nm.startswith("H")
            mask.append(keep)
        return np.asarray(mask, dtype=bool)
    raise ValueError(f"Unknown atom mode: {mode}")


def build_atomgroup(u, spec: SystemSpec):
    if bool(spec.group) == bool(spec.selection):
        raise ValueError(
            f"{spec.name}: provide exactly ONE of --group or --selection. "
            "No analysis group is chosen automatically."
        )

    if spec.group:
        if spec.ndx is None:
            raise ValueError(f"{spec.name}: --group requires --ndx.")
        groups = parse_ndx(spec.ndx)
        actual_name, ids = resolve_ndx_group(groups, spec.group)
        if not ids:
            raise ValueError(f"{spec.name}: index group '{actual_name}' is empty.")
        idx0 = np.asarray(ids, dtype=int) - 1
        if idx0.min() < 0 or idx0.max() >= len(u.atoms):
            raise ValueError(
                f"{spec.name}: index group '{actual_name}' contains atom numbers "
                "outside the topology range."
            )
        ag = u.atoms[idx0]
        print(f"[selection] {spec.name}: index group '{actual_name}' -> {ag.n_atoms} atoms")
    else:
        try:
            ag = u.select_atoms(spec.selection)
        except Exception as exc:
            raise ValueError(
                f"{spec.name}: MDAnalysis selection failed: {spec.selection!r}\n{exc}"
            ) from exc
        print(f"[selection] {spec.name}: MDA selection {spec.selection!r} -> {ag.n_atoms} atoms")

    if ag.n_atoms == 0:
        raise ValueError(f"{spec.name}: analysis selection returned zero atoms.")

    if spec.resids:
        keep_res = set(map(int, spec.resids))
        mask = np.asarray([int(a.resid) in keep_res for a in ag.atoms], dtype=bool)
        ag = ag[mask]
        print(f"[selection] {spec.name}: resid filter {sorted(keep_res)} -> {ag.n_atoms} atoms")

    mask = _atom_mode_mask(ag, spec.atom_mode)
    ag = ag[mask]
    print(f"[selection] {spec.name}: atom mode '{spec.atom_mode}' -> {ag.n_atoms} atoms")

    if ag.n_atoms == 0:
        raise ValueError(
            f"{spec.name}: zero atoms remain after atom-mode/residue filtering."
        )
    return ag


# =============================================================================
# Trajectory preparation
# =============================================================================

def _frame_in_time(ts, start_ns: Optional[float], stop_ns: Optional[float]) -> bool:
    t_ns = float(ts.time) / 1000.0  # GROMACS/MDAnalysis trajectory time is ps
    if start_ns is not None and t_ns < start_ns - 1e-12:
        return False
    if stop_ns is not None and t_ns > stop_ns + 1e-12:
        return False
    return True


def prepare_system(spec: SystemSpec, root: Path) -> PreparedSystem:
    outdir = root / safe_name(spec.name)
    prep = outdir / "00_prepared"
    prep.mkdir(parents=True, exist_ok=True)

    banner(f"PREPARE SYSTEM: {spec.name}")
    u = mda.Universe(str(spec.tpr), str(spec.traj))
    ag = build_atomgroup(u, spec)

    signature = [(int(a.resid), str(a.resname), str(a.name)) for a in ag.atoms]
    indices1 = [int(i) + 1 for i in ag.indices]

    ndx_out = prep / "analysis_selection.ndx"
    write_ndx(ndx_out, "AnalysisSelection", indices1)

    atom_table = prep / "analysis_atoms.tsv"
    with atom_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["selected_order", "topology_atom_1based", "segid", "resid", "resname", "atomname", "mass"])
        for j, a in enumerate(ag.atoms, start=1):
            try:
                mass = float(a.mass)
            except Exception:
                mass = float("nan")
            w.writerow([j, int(a.index) + 1, str(a.segid), int(a.resid), str(a.resname), str(a.name), mass])

    # Collect frame indices first, respecting time and stride.
    chosen_frames: List[int] = []
    times_ns: List[float] = []
    stride = max(1, int(spec.frame_stride))
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0:
            continue
        if not _frame_in_time(ts, spec.start_ns, spec.stop_ns):
            continue
        chosen_frames.append(i)
        times_ns.append(float(ts.time) / 1000.0)

    if not chosen_frames:
        raise ValueError(f"{spec.name}: no trajectory frames selected by the requested time/stride range.")

    selected_xtc = prep / "selected.xtc"
    ref_pdb = prep / "ref_first_frame.pdb"

    # First selected frame becomes the structural reference.
    u.trajectory[chosen_frames[0]]
    ag.write(str(ref_pdb))

    with mda.Writer(str(selected_xtc), n_atoms=ag.n_atoms) as W:
        for frame_i in chosen_frames:
            u.trajectory[frame_i]
            W.write(ag)

    print(f"[prepared] atoms  : {ag.n_atoms}")
    print(f"[prepared] frames : {len(chosen_frames)}")
    print(f"[prepared] time   : {times_ns[0]:.3f} -> {times_ns[-1]:.3f} ns")
    print(f"[prepared] XTC    : {selected_xtc}")
    print(f"[prepared] ref    : {ref_pdb}")

    return PreparedSystem(
        spec=spec,
        outdir=outdir,
        selected_xtc=selected_xtc,
        ref_pdb=ref_pdb,
        selection_ndx=ndx_out,
        atom_table=atom_table,
        n_atoms=ag.n_atoms,
        n_frames=len(chosen_frames),
        first_time_ns=times_ns[0],
        last_time_ns=times_ns[-1],
        signature=signature,
    )


# =============================================================================
# XVG / XPM readers
# =============================================================================

def read_xvg_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@", "&")):
            continue
        p = line.split()
        if len(p) >= 2:
            try:
                xs.append(float(p[0])); ys.append(float(p[1]))
            except ValueError:
                pass
    return np.asarray(xs), np.asarray(ys)


def read_xvg_matrix(path: Path) -> np.ndarray:
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@", "&")):
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            pass
    if not rows:
        return np.empty((0, 0))
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return np.empty((0, 0))
    return np.asarray(rows, dtype=float)


def read_xvg_sets(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    sets: List[List[Tuple[float, float]]] = [[]]
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@")):
            continue
        if line.startswith("&"):
            if sets[-1]:
                sets.append([])
            continue
        p = line.split()
        if len(p) >= 2:
            try:
                sets[-1].append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    out = []
    for s in sets:
        if s:
            arr = np.asarray(s, dtype=float)
            out.append((arr[:, 0], arr[:, 1]))
    return out


def parse_xpm(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Best-effort GROMACS XPM parser -> matrix, xaxis, yaxis."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    xaxis = yaxis = None
    for line in text:
        if "x-axis:" in line:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line.split("x-axis:", 1)[1])
            if nums:
                xaxis = np.asarray([float(x) for x in nums])
        if "y-axis:" in line:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line.split("y-axis:", 1)[1])
            if nums:
                yaxis = np.asarray([float(x) for x in nums])

    header_i = None
    nx = ny = ncolors = cpp = None
    quoted_re = re.compile(r'^\s*"(.*)"\s*,?\s*$')
    for i, line in enumerate(text):
        m = quoted_re.match(line)
        if not m:
            continue
        parts = m.group(1).split()
        if len(parts) == 4 and all(re.fullmatch(r"\d+", p) for p in parts):
            nx, ny, ncolors, cpp = map(int, parts)
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"Could not find XPM header in {path}")

    cmap: Dict[str, float] = {}
    color_lines_start = header_i + 1
    data_start = color_lines_start + ncolors
    for line in text[color_lines_start:data_start]:
        m = quoted_re.match(line.split("/*", 1)[0].strip().rstrip(","))
        if not m:
            # retry with direct quoted content
            qm = re.search(r'"([^"]+)"', line)
            if not qm:
                continue
            content = qm.group(1)
        else:
            content = m.group(1)
        key = content[:cpp]
        comment = ""
        cm = re.search(r'/\*\s*"?([^*\"]+)"?\s*\*/', line)
        if cm:
            comment = cm.group(1).strip()
        num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", comment)
        cmap[key] = float(num.group(0)) if num else np.nan

    rows: List[List[float]] = []
    for line in text[data_start:]:
        qm = re.search(r'"([^"]+)"', line)
        if not qm:
            continue
        s = qm.group(1)
        if len(s) < nx * cpp:
            continue
        vals = [cmap.get(s[j:j + cpp], np.nan) for j in range(0, nx * cpp, cpp)]
        rows.append(vals)
        if len(rows) >= ny:
            break
    if len(rows) != ny:
        raise ValueError(f"Expected {ny} XPM rows but parsed {len(rows)} from {path}")
    mat = np.asarray(rows, dtype=float)
    # GROMACS XPM is usually top-to-bottom; flip for Cartesian plotting.
    mat = mat[::-1, :]
    if yaxis is not None and len(yaxis) == mat.shape[0]:
        yaxis = yaxis[::-1]
    return mat, xaxis, yaxis


# =============================================================================
# Plot helpers
# =============================================================================

def savefig(fig, base: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scree(eig_xvg: Path, outbase: Path, n_show: int, dpi: int) -> Dict[str, float]:
    x, y = read_xvg_xy(eig_xvg)
    if len(y) == 0:
        return {}
    vals = np.clip(y, 0, None)
    total = vals.sum()
    cumulative = np.cumsum(vals) / total * 100 if total > 0 else np.zeros_like(vals)
    n = min(n_show, len(vals))
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(x[:n], vals[:n], marker="o", linewidth=1.5)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue (nm$^2$)")
    ax1.set_xticks(x[:n].astype(int) if np.allclose(x[:n], np.round(x[:n])) else x[:n])
    ax2 = ax1.twinx()
    ax2.plot(x[:n], cumulative[:n], marker="s", linewidth=1.3, linestyle="--")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)
    ax1.set_title("PCA Eigenvalues and Cumulative Variance")
    savefig(fig, outbase, dpi)
    metrics = {f"pc{i+1}_variance_pct": float(vals[i] / total * 100) if total > 0 else 0.0 for i in range(min(10, len(vals)))}
    for ncomp in (2, 3, 5, 10):
        if len(vals) >= ncomp:
            metrics[f"cumulative_{ncomp}_pct"] = float(cumulative[ncomp - 1])
    return metrics


def plot_cosine(cc_xvg: Path, outbase: Path, dpi: int) -> Dict[str, float]:
    x, y = read_xvg_xy(cc_xvg)
    if len(y) == 0:
        return {}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, y)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cosine content")
    ax.set_ylim(0, max(1.0, float(np.nanmax(y)) * 1.1))
    ax.set_title("Cosine Content of Principal Components")
    savefig(fig, outbase, dpi)
    return {f"pc{int(round(px))}_cosine": float(py) for px, py in zip(x, y)}


def plot_projection_timeseries(proj_xvg: Path, outbase: Path, dpi: int) -> None:
    sets = read_xvg_sets(proj_xvg)
    if not sets:
        # fallback: matrix columns: time pc1 pc2 ...
        arr = read_xvg_matrix(proj_xvg)
        if arr.shape[1] >= 2:
            sets = [(arr[:, 0], arr[:, i]) for i in range(1, arr.shape[1])]
    if not sets:
        return
    n = len(sets)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.0, rows * 2.7)), squeeze=False)
    for i, (x, y) in enumerate(sets):
        ax = axes.flat[i]
        # GROMACS time from selected XTC is usually ps. Convert if clearly large.
        tx = x / 1000.0 if np.nanmax(x) > 500 else x
        ax.plot(tx, y, linewidth=0.8)
        ax.set_title(f"PC{i+1}")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Projection")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Projection of the Trajectory on Principal Components", y=1.005, fontsize=14)
    savefig(fig, outbase, dpi)


def read_2d_projection(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = read_xvg_matrix(path)
    if arr.size == 0:
        return np.array([]), np.array([])
    # gmx anaeig -2d typically writes two data columns. If a time column appears,
    # use the final two columns.
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    if arr.shape[1] >= 3:
        return arr[:, -2], arr[:, -1]
    return np.array([]), np.array([])


def plot_pca_scatter(path: Path, pair: Tuple[int, int], outbase: Path, dpi: int, title_prefix: str = "") -> None:
    x, y = read_2d_projection(path)
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    c = np.arange(len(x))
    sc = ax.scatter(x, y, c=c, s=9, cmap="viridis", alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Frame order")
    ax.scatter([np.mean(x)], [np.mean(y)], s=70, marker="*", edgecolor="black", linewidth=0.6)
    ax.set_xlabel(f"PC{pair[0]}")
    ax.set_ylabel(f"PC{pair[1]}")
    ax.set_title(f"{title_prefix} PCA Projection: PC{pair[0]} vs PC{pair[1]}".strip())
    savefig(fig, outbase, dpi)



def read_projection_first3(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer XVG multi-set semantics: gmx anaeig -proj commonly writes one set per PC.
    sets = read_xvg_sets(path)
    if len(sets) >= 3:
        n = min(len(sets[0][1]), len(sets[1][1]), len(sets[2][1]))
        return sets[0][1][:n], sets[1][1][:n], sets[2][1][:n]
    arr = read_xvg_matrix(path)
    if arr.size == 0 or arr.shape[1] < 3:
        return np.array([]), np.array([]), np.array([])
    # Matrix fallback: when there are >=4 columns, first is normally time and
    # columns 1-3 are PC1-PC3. With exactly 3 columns, use them directly.
    cols = arr[:, 1:4] if arr.shape[1] >= 4 else arr[:, :3]
    return cols[:,0], cols[:,1], cols[:,2]


def plot_pca_3d_from_proj(proj: Path, outbase: Path, dpi: int, title: str) -> bool:
    x, y, z = read_projection_first3(proj)
    if len(x) == 0:
        return False
    fig = plt.figure(figsize=(8.4, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    c = np.arange(len(x))
    sc = ax.scatter(x, y, z, c=c, s=8, cmap="viridis", alpha=0.65, linewidths=0)
    fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.70, label="Frame order")
    ax.scatter([np.mean(x)], [np.mean(y)], [np.mean(z)], marker="*", s=130, edgecolor="black")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-58)
    savefig(fig, outbase, dpi)
    return True


def plot_common_pca_3d_overlay(proj1: Path, proj2: Path, name1: str, name2: str,
                               outbase: Path, dpi: int) -> Dict[str, float]:
    x1,y1,z1 = read_projection_first3(proj1); x2,y2,z2 = read_projection_first3(proj2)
    if len(x1)==0 or len(x2)==0:
        return {}
    fig=plt.figure(figsize=(9.0,7.2)); ax=fig.add_subplot(111, projection='3d')
    ax.scatter(x1,y1,z1,s=9,alpha=.40,label=name1)
    ax.scatter(x2,y2,z2,s=9,alpha=.40,label=name2)
    c1=np.array([np.mean(x1),np.mean(y1),np.mean(z1)])
    c2=np.array([np.mean(x2),np.mean(y2),np.mean(z2)])
    ax.scatter(*c1,marker='*',s=150,edgecolor='black'); ax.scatter(*c2,marker='*',s=150,edgecolor='black')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title(f'Common PCA 3D: {name1} vs {name2}')
    ax.legend(); ax.view_init(elev=22,azim=-58)
    savefig(fig,outbase,dpi)
    return {'centroid_separation_3D_PC1_PC2_PC3':float(np.linalg.norm(c1-c2))}

def plot_xpm_heatmap(xpm: Path, outbase: Path, title: str, cbar_label: str, dpi: int, cmap: str = "viridis") -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    try:
        mat, xa, ya = parse_xpm(xpm)
    except Exception as exc:
        print(f"[plot][skip] Could not parse {xpm.name}: {exc}")
        return None
    fig, ax = plt.subplots(figsize=(7.3, 6.0))
    if xa is not None and ya is not None and len(xa) == mat.shape[1] and len(ya) == mat.shape[0]:
        im = ax.imshow(mat, origin="lower", aspect="auto", extent=[xa.min(), xa.max(), ya.min(), ya.max()], cmap=cmap)
    else:
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    savefig(fig, outbase, dpi)
    return mat, xa, ya


def plot_fel(xpm: Path, pair: Tuple[int, int], outdir: Path, prefix: str, dpi: int) -> Dict[str, float]:
    try:
        z, xa, ya = parse_xpm(xpm)
    except Exception as exc:
        print(f"[FEL][plot-skip] {xpm.name}: {exc}")
        return {}
    finite = np.isfinite(z)
    if not finite.any():
        return {}
    zplot = np.array(z, copy=True)
    # replace NaN for visual only
    zplot[~finite] = np.nanmax(zplot[finite])
    if xa is None or len(xa) != z.shape[1]:
        xa = np.arange(z.shape[1], dtype=float)
    if ya is None or len(ya) != z.shape[0]:
        ya = np.arange(z.shape[0], dtype=float)
    X, Y = np.meshgrid(xa, ya)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    cf = ax.contourf(X, Y, zplot, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Free energy (kJ/mol)")
    ax.set_xlabel(f"PC{pair[0]}")
    ax.set_ylabel(f"PC{pair[1]}")
    ax.set_title(f"Free Energy Landscape: PC{pair[0]} vs PC{pair[1]}")
    savefig(fig, outdir / f"{prefix}_2D", dpi)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        step_x = max(1, int(math.ceil(zplot.shape[1] / 100)))
        step_y = max(1, int(math.ceil(zplot.shape[0] / 100)))
        ax.plot_surface(X[::step_y, ::step_x], Y[::step_y, ::step_x], zplot[::step_y, ::step_x], cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel(f"PC{pair[0]}")
        ax.set_ylabel(f"PC{pair[1]}")
        ax.set_zlabel("G (kJ/mol)")
        ax.set_title(f"3D Free Energy Landscape: PC{pair[0]} vs PC{pair[1]}")
        savefig(fig, outdir / f"{prefix}_3D", dpi)
    except Exception as exc:
        print(f"[FEL][3D-skip] {exc}")

    idx = np.nanargmin(z)
    iy, ix = np.unravel_index(idx, z.shape)
    return {
        "fel_min_energy_kjmol": float(z[iy, ix]),
        f"fel_min_pc{pair[0]}": float(xa[ix]),
        f"fel_min_pc{pair[1]}": float(ya[iy]),
        "fel_max_finite_kjmol": float(np.nanmax(z[finite])),
    }


def plot_fel_from_projection(proj2d: Path, pair: Tuple[int, int], outdir: Path,
                             prefix: str, dpi: int, temperature: float,
                             bins: int = 80) -> Dict[str, float]:
    """Guaranteed FEL plotting fallback directly from the 2D PCA projection.

    Uses G = -R*T*ln(P/Pmax). This is used when GROMACS XPM parsing fails, and
    also provides a consistent PNG/SVG output for every PCA pair.
    """
    x, y = read_2d_projection(proj2d)
    if len(x) < 10 or len(y) < 10:
        print(f"[FEL][warning] Too few projection points in {proj2d}")
        return {}
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    P = H / max(float(H.sum()), 1.0)
    positive = P > 0
    G = np.full_like(P, np.nan, dtype=float)
    if not np.any(positive):
        return {}
    R_kJ = 0.00831446261815324
    pmax = float(np.max(P[positive]))
    G[positive] = -R_kJ * float(temperature) * np.log(P[positive] / pmax)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Z = G.T
    X, Y = np.meshgrid(xc, yc)
    finite = np.isfinite(Z)
    if not np.any(finite):
        return {}
    zfill = np.array(Z, copy=True)
    zfill[~finite] = np.nanmax(zfill[finite])

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    cf = ax.contourf(X, Y, zfill, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Free energy (kJ/mol)")
    ax.set_xlabel(f"PC{pair[0]}")
    ax.set_ylabel(f"PC{pair[1]}")
    ax.set_title(f"Free Energy Landscape: PC{pair[0]} vs PC{pair[1]}")
    savefig(fig, outdir / f"{prefix}_2D", dpi)

    try:
        fig = plt.figure(figsize=(8.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, zfill, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel(f"PC{pair[0]}")
        ax.set_ylabel(f"PC{pair[1]}")
        ax.set_zlabel("G (kJ/mol)")
        ax.set_title(f"3D Free Energy Landscape: PC{pair[0]} vs PC{pair[1]}")
        savefig(fig, outdir / f"{prefix}_3D", dpi)
    except Exception as exc:
        print(f"[FEL][3D-warning] {exc}")

    iy, ix = np.unravel_index(np.nanargmin(Z), Z.shape)
    return {
        "fel_min_energy_kjmol": float(Z[iy, ix]),
        f"fel_min_pc{pair[0]}": float(xc[ix]),
        f"fel_min_pc{pair[1]}": float(yc[iy]),
        "fel_max_finite_kjmol": float(np.nanmax(Z[finite])),
        "fel_method": "direct_histogram_-RTlnP",
    }


def guaranteed_fel(gmx_files: Dict[str, Path], proj2d: Path, pair: Tuple[int, int],
                   outdir: Path, prefix: str, dpi: int, temperature: float) -> Dict[str, float]:
    """Prefer GROMACS Gibbs XPM; guarantee plots with direct projection fallback."""
    m: Dict[str, float] = {}
    if gmx_files.get("gibbs") and Path(gmx_files["gibbs"]).exists():
        m = plot_fel(Path(gmx_files["gibbs"]), pair, outdir, prefix, dpi)
    png = outdir / f"{prefix}_2D.png"
    if not png.exists() or not m:
        print(f"[FEL] Using direct -RT ln(P) fallback for PC{pair[0]}-PC{pair[1]}")
        m = plot_fel_from_projection(proj2d, pair, outdir, prefix, dpi, temperature)
    return m


def read_pdb_models_ca(path: Path) -> List[np.ndarray]:
    """Read CA coordinates from each MODEL in a PDB; supports single-model PDBs."""
    models: List[List[List[float]]] = []
    current: List[List[float]] = []
    saw_model = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            saw_model = True
            if current:
                models.append(current); current = []
        elif rec in ("ATOM", "HETATM") and line[12:16].strip() == "CA":
            try:
                current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
        elif rec == "ENDMDL":
            if current:
                models.append(current); current = []
    if current:
        models.append(current)
    return [np.asarray(m, dtype=float) for m in models if m]



def read_pdb_ca_labels(path: Path) -> List[str]:
    labels=[]; in_first=False; saw_model=False
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        rec=line[:6].strip()
        if rec=='MODEL':
            if saw_model: break
            saw_model=True; in_first=True; continue
        if rec=='ENDMDL' and saw_model: break
        if rec in ('ATOM','HETATM') and line[12:16].strip()=='CA':
            resn=line[17:20].strip(); chain=line[21:22].strip(); resid=line[22:26].strip()
            labels.append(f"{resn}{resid}{chain if chain else ''}")
    return labels


def porcupine_top_motions(extreme_pdb: Path, top_n: int=10) -> List[Tuple[str,float]]:
    dat=porcupine_data(extreme_pdb,max_arrows=100000)
    if dat is None: return []
    xyz,disp,keep=dat; labels=read_pdb_ca_labels(extreme_pdb)
    mag=np.linalg.norm(disp,axis=1); order=np.argsort(mag)[::-1][:min(top_n,len(mag))]
    return [(labels[i] if i<len(labels) else f"CA_{i+1}",float(mag[i])) for i in order]

def porcupine_data(extreme_pdb: Path, max_arrows: int = 70) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    models = read_pdb_models_ca(extreme_pdb)
    if len(models) < 2:
        print(f"[porcupine][warning] Need >=2 PDB models: {extreme_pdb}")
        return None
    a, b = models[0], models[-1]
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    d = b - a
    mag = np.linalg.norm(d, axis=1)
    # Keep informative arrows while preventing an unreadable 353-arrow figure.
    positive = np.where(mag > 1e-8)[0]
    if len(positive) == 0:
        return None
    if len(positive) > max_arrows:
        keep = positive[np.argsort(mag[positive])[-max_arrows:]]
        keep = np.sort(keep)
    else:
        keep = positive
    return a, d, keep


def _equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = np.nanmin(xyz, axis=0); maxs = np.nanmax(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.55, 1.0)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)


def plot_porcupine(extreme_pdb: Path, outbase: Path, title: str, dpi: int) -> bool:
    dat = porcupine_data(extreme_pdb)
    if dat is None:
        return False
    xyz, disp, keep = dat
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], linewidth=1.0, alpha=0.65)
    ax.quiver(xyz[keep,0], xyz[keep,1], xyz[keep,2],
              disp[keep,0], disp[keep,1], disp[keep,2],
              normalize=False, arrow_length_ratio=0.25, linewidth=0.8)
    labels = read_pdb_ca_labels(extreme_pdb)
    mags = np.linalg.norm(disp, axis=1)
    for i in np.argsort(mags)[::-1][:min(8, len(mags))]:
        if i < len(labels):
            ax.text(xyz[i,0], xyz[i,1], xyz[i,2], labels[i], fontsize=6)
    ax.set_title(title)
    ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
    _equal_3d_axes(ax, xyz)
    ax.view_init(elev=20, azim=-60)
    savefig(fig, outbase, dpi)
    return True


def plot_porcupine_comparison(extreme_a: Path, extreme_b: Path, outbase: Path,
                              name_a: str, name_b: str, pc: int, dpi: int,
                              basis_label: str = "Common PCA basis") -> bool:
    da, db = porcupine_data(extreme_a), porcupine_data(extreme_b)
    if da is None or db is None:
        return False
    fig = plt.figure(figsize=(15.0, 6.8))
    for pos, dat, name in ((1, da, name_a), (2, db, name_b)):
        xyz, disp, keep = dat
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], linewidth=1.0, alpha=0.65)
        ax.quiver(xyz[keep,0], xyz[keep,1], xyz[keep,2],
                  disp[keep,0], disp[keep,1], disp[keep,2],
                  normalize=False, arrow_length_ratio=0.25, linewidth=0.8)
        labels = read_pdb_ca_labels(extreme_a if pos == 1 else extreme_b)
        mags = np.linalg.norm(disp, axis=1)
        for i in np.argsort(mags)[::-1][:min(6, len(mags))]:
            if i < len(labels): ax.text(xyz[i,0], xyz[i,1], xyz[i,2], labels[i], fontsize=6)
        ax.set_title(f"{name} — PC{pc}")
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
        _equal_3d_axes(ax, xyz)
        ax.view_init(elev=20, azim=-60)
    fig.suptitle(f"Porcupine comparison — PC{pc} ({basis_label})", fontsize=14)
    savefig(fig, outbase, dpi)
    return True



def _projection_component(path: Path, column: int) -> np.ndarray:
    """Return a zero-based numeric column from an XVG data table."""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        t = line.strip()
        if not t or t.startswith(("#", "@")):
            continue
        try:
            vals = [float(x) for x in t.split()]
        except ValueError:
            continue
        if len(vals) > column:
            rows.append(vals[column])
    return np.asarray(rows, dtype=float)


def _robust_span(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    q05, q95 = np.percentile(v, [5.0, 95.0])
    return float(q95 - q05)


def _kabsch_transform(mobile: np.ndarray, target: np.ndarray):
    """Return R,t such that mobile @ R + t best fits target (row-vector convention)."""
    mobile = np.asarray(mobile, float); target = np.asarray(target, float)
    cm = mobile.mean(axis=0); ct = target.mean(axis=0)
    x = mobile - cm; y = target - ct
    u, _, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    t = ct - cm @ r
    return r, t


def _aligned_extreme_motion(extreme_pdb: Path, target_first: Optional[np.ndarray] = None):
    models = read_pdb_models_ca(extreme_pdb)
    if len(models) < 2:
        raise RuntimeError(f"Need >=2 models in {extreme_pdb}")
    a = np.asarray(models[0], float); b = np.asarray(models[-1], float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if target_first is None:
        target_first = a.copy()
    else:
        target_first = np.asarray(target_first, float)[:n]
    r, t = _kabsch_transform(a, target_first)
    aa = a @ r + t
    bb = b @ r + t
    return aa, bb - aa


def _global_vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, float).ravel(); y = np.asarray(b, float).ravel()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else float("nan")


def _normalize_motion_rms(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, float)
    rms = float(np.sqrt(np.mean(np.sum(d*d, axis=1))))
    return d / rms if rms > 0 else d.copy()


def _shared_arrow_indices(d1: np.ndarray, d2: np.ndarray, max_arrows: int = 70) -> np.ndarray:
    score = np.maximum(np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1))
    positive = np.where(score > 1e-10)[0]
    if len(positive) > max_arrows:
        positive = positive[np.argsort(score[positive])[-max_arrows:]]
    return np.sort(positive)


def rigorous_common_sampling_plot(extreme_common: Path, proj_a: Path, proj_b: Path,
                                  proj_column: int,
                                  name_a: str, name_b: str, pc: int,
                                  outbase: Path, dpi: int) -> Dict[str, float]:
    """
    Common-basis comparison: SAME mode shape/direction for both systems.
    Only robust sampling amplitude (P95-P05 of projection) is allowed to differ.
    Arrow scale is shared between panels.
    """
    xyz, d = _aligned_extreme_motion(extreme_common)
    d = _normalize_motion_rms(d)
    span_a = _robust_span(_projection_component(proj_a, proj_column))
    span_b = _robust_span(_projection_component(proj_b, proj_column))
    mx = max(span_a, span_b) if np.isfinite(span_a) and np.isfinite(span_b) else 1.0
    fa = span_a / mx if mx > 0 else 0.0
    fb = span_b / mx if mx > 0 else 0.0
    # Shared display multiplier: improves visibility without changing A/B amplitude ratio.
    display_gain = 5.0
    da, db = d * fa * display_gain, d * fb * display_gain
    keep = _shared_arrow_indices(da, db, max_arrows=45)

    fig = plt.figure(figsize=(16, 7.2))
    for pos, disp, nm, span in ((1, da, name_a, span_a), (2, db, name_b, span_b)):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], linewidth=1.0, alpha=0.65)
        ax.quiver(xyz[keep,0], xyz[keep,1], xyz[keep,2],
                  disp[keep,0], disp[keep,1], disp[keep,2],
                  normalize=False, arrow_length_ratio=0.30, linewidth=1.15)
        labels = read_pdb_ca_labels(extreme_common)
        top = keep[np.argsort(np.linalg.norm(disp[keep], axis=1))[-8:]] if len(keep) else []
        for j, i in enumerate(top):
            if i < len(labels):
                zoff = (j - (len(top)-1)/2.0) * 1.4
                ax.text(xyz[i,0], xyz[i,1], xyz[i,2] + zoff,
                        labels[i], fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"))
        ax.set_title(f"{nm} — PC{pc}\nP95-P05 projection span = {span:.4g}")
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
        _equal_3d_axes(ax, xyz); ax.view_init(elev=20, azim=-60)
    fig.suptitle(f"Common-basis PC{pc}: SAME direction; arrow LENGTH = relative sampling amplitude", fontsize=14)
    savefig(fig, outbase, dpi)
    return {
        "PC": pc, f"{name_a}_P95_P05": span_a, f"{name_b}_P95_P05": span_b,
        f"{name_a}_relative_amplitude": fa, f"{name_b}_relative_amplitude": fb,
        f"{name_a}_over_{name_b}_span_ratio": (span_a/span_b if span_b > 0 else float("nan")),
    }


def rigorous_individual_direction_plot(extreme_a: Path, extreme_b: Path,
                                       name_a: str, name_b: str, pc: int,
                                       outbase: Path, residue_csv: Path,
                                       dpi: int) -> Dict[str, float]:
    """
    Individual-PCA comparison: compare mode SHAPE/DIRECTION.
    Kabsch-align first states, remove arbitrary eigenvector sign, RMS-normalize
    each displacement field, then compare residue by residue on a shared scale.
    """
    ma = read_pdb_models_ca(extreme_a)
    if len(ma) < 2:
        raise RuntimeError(f"Need >=2 models in {extreme_a}")
    ref = np.asarray(ma[0], float)
    xa, da = _aligned_extreme_motion(extreme_a, ref)
    xb, db = _aligned_extreme_motion(extreme_b, ref)
    n = min(len(xa), len(xb), len(da), len(db))
    xa, da, xb, db = xa[:n], da[:n], xb[:n], db[:n]

    da = _normalize_motion_rms(da); db = _normalize_motion_rms(db)
    raw_cos = _global_vector_cosine(da, db)
    sign_flipped = bool(np.isfinite(raw_cos) and raw_cos < 0)
    if sign_flipped:
        db = -db
    mode_cos = _global_vector_cosine(da, db)

    ma_mag = np.linalg.norm(da, axis=1); mb_mag = np.linalg.norm(db, axis=1)
    dot = np.sum(da*db, axis=1)
    den = ma_mag*mb_mag
    local_cos = np.divide(dot, den, out=np.full(n, np.nan), where=den > 1e-12)
    local_cos = np.clip(local_cos, -1.0, 1.0)
    angle = np.degrees(np.arccos(local_cos))
    mag_diff = mb_mag - ma_mag
    vec_diff = np.linalg.norm(db-da, axis=1)

    labels = read_pdb_ca_labels(extreme_a)
    with residue_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["rank","atom_index","label",f"{name_a}_norm_magnitude",f"{name_b}_norm_magnitude",
                  "motion_angle_deg","magnitude_difference_B_minus_A","vector_difference"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        order = np.argsort(vec_diff)[::-1]
        for rank, i in enumerate(order, 1):
            w.writerow({
                "rank": rank, "atom_index": i+1,
                "label": labels[i] if i < len(labels) else f"CA_{i+1}",
                f"{name_a}_norm_magnitude": float(ma_mag[i]),
                f"{name_b}_norm_magnitude": float(mb_mag[i]),
                "motion_angle_deg": float(angle[i]) if np.isfinite(angle[i]) else "",
                "magnitude_difference_B_minus_A": float(mag_diff[i]),
                "vector_difference": float(vec_diff[i]),
            })

    # These vectors are RMS-normalized by design: this figure compares DIRECTION/SHAPE,
    # not physical motion amplitude. Use a shared display gain only for readability.
    display_gain = 4.0
    da_plot, db_plot = da * display_gain, db * display_gain
    keep = _shared_arrow_indices(da_plot, db_plot, max_arrows=45)
    fig = plt.figure(figsize=(16, 7.2))
    for pos, xyz, disp, nm in ((1, xa, da_plot, name_a), (2, xb, db_plot, name_b)):
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], linewidth=1.0, alpha=0.65)
        ax.quiver(xyz[keep,0], xyz[keep,1], xyz[keep,2],
                  disp[keep,0], disp[keep,1], disp[keep,2],
                  normalize=False, arrow_length_ratio=0.30, linewidth=1.15)
        top = np.argsort(vec_diff)[::-1][:8]
        for j, i in enumerate(top):
            if i < len(labels):
                zoff = (j - (len(top)-1)/2.0) * 1.4
                ax.text(xyz[i,0], xyz[i,1], xyz[i,2] + zoff,
                        labels[i], fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"))
        ax.set_title(f"{nm} — individual PC{pc}")
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
        _equal_3d_axes(ax, ref); ax.view_init(elev=20, azim=-60)
    fig.suptitle(f"Individual-PCA PC{pc}: DIRECTION/SHAPE comparison (lengths normalized)\n"
                 f"global mode cosine = {mode_cos:.3f}", fontsize=13)
    savefig(fig, outbase, dpi)

    return {
        "PC": pc, "raw_mode_cosine_before_sign_alignment": raw_cos,
        "sign_flipped_for_comparison": sign_flipped,
        "aligned_normalized_mode_cosine": mode_cos,
        "median_residue_motion_angle_deg": float(np.nanmedian(angle)),
        "mean_residue_motion_angle_deg": float(np.nanmean(angle)),
        "median_vector_difference": float(np.nanmedian(vec_diff)),
        "max_vector_difference": float(np.nanmax(vec_diff)),
    }


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def write_comparison_html(common: Path, name_a: str, name_b: str,
                          metrics_rows: List[Dict[str, object]],
                          ev: Dict[str, object],
                          common_sampling: List[Dict[str, object]],
                          individual_direction: List[Dict[str, object]]) -> Path:
    import html
    def relimg(p):
        p = common / p
        return f'<img src="{html.escape(str(p.relative_to(common)))}" style="max-width:100%;height:auto">'
    def table(rows):
        if not rows: return "<p>No data.</p>"
        keys=[]
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        h="<table><thead><tr>"+''.join(f"<th>{html.escape(str(k))}</th>" for k in keys)+"</tr></thead><tbody>"
        for r in rows:
            h+="<tr>"+''.join(f"<td>{html.escape(str(r.get(k,'')))}</td>" for k in keys)+"</tr>"
        return h+"</tbody></table>"

    imgs = []
    for stem in ["OVERLAY_PC1_PC2","OVERLAY_PC1_PC3","OVERLAY_PC2_PC3",
                 "OVERLAY_PC1_PC2_PC3_3D","common_scree_plot"]:
        if (common/f"{stem}.png").exists(): imgs.append(relimg(f"{stem}.png"))
    por = common/"PORCUPINE_COMMON_BASIS"
    porimgs=[]
    for pc in (1,2,3):
        for stem in [f"COMMON_SAMPLING_{safe_name(name_a)}_vs_{safe_name(name_b)}_PC{pc}",
                     f"INDIVIDUAL_DIRECTION_{safe_name(name_a)}_vs_{safe_name(name_b)}_PC{pc}"]:
            if (por/f"{stem}.png").exists():
                porimgs.append(relimg(f"PORCUPINE_COMMON_BASIS/{stem}.png"))
    felimgs=[]
    for nm in (name_a,name_b):
        d=common/f"FEL_{safe_name(nm)}_common_basis"
        if d.exists():
            for p in sorted(d.glob("*.png")):
                felimgs.append(relimg(str(p.relative_to(common))))
    css="""body{font-family:Arial,sans-serif;max-width:1250px;margin:30px auto;line-height:1.45}
    h1,h2{margin-top:28px} table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{border:1px solid #bbb;padding:6px;text-align:left} th{background:#eee}
    img{margin:10px 0 22px 0;border:1px solid #ddd}.note{padding:12px;background:#f4f4f4}"""
    body=f"""<!doctype html><html><head><meta charset="utf-8"><title>PCA/FEL comparison report</title>
    <style>{css}</style></head><body>
    <h1>PCA/FEL comparison: {html.escape(name_a)} vs {html.escape(name_b)}</h1>
    <p>Script version: {html.escape(SCRIPT_VERSION)}</p>
    <div class="note"><b>Porcupine interpretation:</b> Common-basis figures compare sampling amplitude
    along the same shared eigenvector. Individual-PCA figures compare mode shape/direction only after
    Kabsch alignment, arbitrary sign alignment, and RMS normalization. PCA/FEL separation alone does
    not prove increased thermodynamic stability or causality.</div>
    <h2>Conformational-space metrics</h2>{table(metrics_rows)}
    <h2>PCA figures</h2>{''.join(imgs)}
    <h2>Common-basis FEL</h2>{''.join(felimgs)}
    <h2>Common-basis Porcupine sampling amplitude</h2>{table(common_sampling)}
    <h2>Individual-PCA directional comparison</h2>{table(individual_direction)}
    {''.join(porimgs)}
    <h2>Eigenvector/RMSIP metrics</h2><pre>{html.escape(json.dumps(ev,indent=2,default=str))}</pre>
    </body></html>"""
    out=common/"PCA_FEL_COMPARISON_REPORT.html"
    out.write_text(body,encoding="utf-8")
    return out

# =============================================================================
# GROMACS PCA / FEL engine
# =============================================================================

def gmx_covar(
    gmx: str,
    ref: Path,
    xtc: Path,
    outdir: Path,
    *,
    n_pcs: int,
    mass_weighted: bool,
    ref_mode_first: bool,
    begin_ns: Optional[float] = None,
    end_ns: Optional[float] = None,
    tag: str = "all",
) -> Tuple[Path, Path, Path, Path]:
    eig = outdir / f"eigenvalues_{tag}.xvg"
    vec = outdir / f"eigenvectors_{tag}.trr"
    avg = outdir / f"average_{tag}.pdb"
    xpm = outdir / f"covariance_{tag}.xpm"
    log = outdir / f"covar_{tag}.log"
    cmd = [gmx, "covar", "-s", ref, "-f", xtc, "-o", eig, "-v", vec,
           "-av", avg, "-xpm", xpm, "-l", log, "-last", str(n_pcs)]
    if mass_weighted:
        cmd += ["-mwa"]
    if ref_mode_first:
        cmd += ["-ref"]
    if begin_ns is not None:
        cmd += ["-b", str(begin_ns)]
    if end_ns is not None:
        cmd += ["-e", str(end_ns)]
    if begin_ns is not None or end_ns is not None:
        cmd += ["-tu", "ns"]
    run_cmd(cmd, stdin="0\n0\n", cwd=outdir, label=f"gmx covar ({tag})", log_path=outdir / f"run_covar_{tag}.log")
    return eig, vec, avg, xpm


def gmx_projection(
    gmx: str,
    ref: Path,
    xtc: Path,
    eig: Path,
    vec: Path,
    out: Path,
    *,
    first: int,
    last: int,
    mode: str,
) -> None:
    cmd = [gmx, "anaeig", "-s", ref, "-f", xtc, "-v", vec, "-eig", eig,
           "-first", str(first), "-last", str(last)]
    if mode == "proj":
        cmd += ["-proj", out]
    elif mode == "2d":
        cmd += ["-2d", out]
    elif mode == "3d":
        cmd += ["-3d", out]
    elif mode == "extr":
        cmd += ["-extr", out, "-nframes", "30"]
    else:
        raise ValueError(mode)
    run_cmd(cmd, stdin="0\n0\n", cwd=out.parent, label=f"gmx anaeig {mode} PC{first}-PC{last}")


def gmx_cosine(gmx: str, proj: Path, out: Path, n_pcs: int) -> None:
    run_cmd([gmx, "analyze", "-f", proj, "-n", str(n_pcs), "-cc", out], cwd=out.parent, label="gmx analyze cosine content")


def gmx_sham(gmx: str, proj2d: Path, outdir: Path, pair: Tuple[int, int], temperature: float) -> Dict[str, Path]:
    a, b = pair
    tag = f"{a}{b}"
    files = {
        "gibbs": outdir / f"gibbs_{tag}.xpm",
        "dist": outdir / f"energy_distribution_{tag}.xvg",
        "histo": outdir / f"energy_histogram_{tag}.xvg",
        "bin": outdir / f"bindex_{tag}.ndx",
        "prob": outdir / f"probability_{tag}.xpm",
        "enthalpy": outdir / f"enthalpy_{tag}.xpm",
        "entropy": outdir / f"entropy_{tag}.xpm",
        "gibbs3d": outdir / f"gibbs3_{tag}.pdb",
        "log": outdir / f"sham_{tag}.log",
    }
    cmd = [gmx, "sham", "-f", proj2d, "-notime", "-tsham", str(temperature),
           "-ls", files["gibbs"], "-dist", files["dist"], "-histo", files["histo"],
           "-bin", files["bin"], "-lp", files["prob"], "-lsh", files["enthalpy"],
           "-lss", files["entropy"], "-ls3", files["gibbs3d"], "-g", files["log"]]
    run_cmd(cmd, cwd=outdir, label=f"gmx sham FEL PC{a}-PC{b}", log_path=outdir / f"run_sham_{tag}.log")
    return files


def convergence_endpoints(last_ns: float, step_ns: float, start_ns: float = 0.0) -> List[float]:
    if step_ns <= 0 or last_ns <= start_ns:
        return []
    vals = []
    t = start_ns + step_ns
    while t < last_ns - 1e-8:
        vals.append(round(t, 6))
        t += step_ns
    vals.append(round(last_ns, 6))
    # unique, sorted
    return sorted(set(vals))


def run_single_analysis(
    prep: PreparedSystem,
    *,
    gmx: str,
    temperature: float,
    n_pcs: int,
    cosine_pcs: int,
    convergence_step_ns: float,
    mass_weighted: bool,
    ref_mode_first: bool,
    dpi: int,
    make_fel: bool = True,
) -> Dict[str, object]:
    root = prep.outdir
    pca = root / "01_PCA"
    conv = root / "02_PCA_convergence"
    fel = root / "03_FEL"
    modes = root / "04_modes"
    plots = root / "05_plots"
    for d in (pca, conv, fel, modes, plots):
        d.mkdir(parents=True, exist_ok=True)

    banner(f"PCA: {prep.spec.name}")
    eig, vec, avg, cov_xpm = gmx_covar(
        gmx, prep.ref_pdb, prep.selected_xtc, pca,
        n_pcs=n_pcs,
        mass_weighted=mass_weighted,
        ref_mode_first=ref_mode_first,
    )

    metrics: Dict[str, object] = {
        "name": prep.spec.name,
        "n_atoms": prep.n_atoms,
        "n_frames": prep.n_frames,
        "first_time_ns": prep.first_time_ns,
        "last_time_ns": prep.last_time_ns,
        "selection_group": prep.spec.group,
        "selection_mda": prep.spec.selection,
        "atom_mode": prep.spec.atom_mode,
        "resids": prep.spec.resids,
    }

    metrics.update(plot_scree(eig, plots / "scree_plot", n_pcs, dpi))
    plot_xpm_heatmap(cov_xpm, plots / "covariance_matrix", "Covariance Matrix", "Covariance", dpi, cmap="coolwarm")

    # Projections + cosine content
    ncc = min(cosine_pcs, n_pcs)
    proj_first = pca / f"projection_first_{ncc}_PCs.xvg"
    gmx_projection(gmx, prep.ref_pdb, prep.selected_xtc, eig, vec, proj_first, first=1, last=ncc, mode="proj")
    cc = pca / f"cosine_content_first_{ncc}.xvg"
    gmx_cosine(gmx, proj_first, cc, ncc)
    metrics.update(plot_cosine(cc, plots / "cosine_content", dpi))
    plot_projection_timeseries(proj_first, plots / "PC_timeseries", dpi)
    if n_pcs >= 3:
        plot_pca_3d_from_proj(proj_first, plots / "PCA_PC1_PC2_PC3_3D", dpi,
                              f"{prep.spec.name} PCA 3D Projection: PC1/PC2/PC3")

    # 2D / 3D projections
    proj_paths: Dict[Tuple[int, int], Path] = {}
    for pair in PC_PAIRS:
        if max(pair) > n_pcs:
            continue
        out = pca / f"proj_PC{pair[0]}_PC{pair[1]}.xvg"
        gmx_projection(gmx, prep.ref_pdb, prep.selected_xtc, eig, vec, out, first=pair[0], last=pair[1], mode="2d")
        proj_paths[pair] = out
        plot_pca_scatter(out, pair, plots / f"PCA_PC{pair[0]}_PC{pair[1]}", dpi, title_prefix=prep.spec.name)

    if n_pcs >= 3:
        p3 = pca / "proj_PC1_PC2_PC3.pdb"
        gmx_projection(gmx, prep.ref_pdb, prep.selected_xtc, eig, vec, p3, first=1, last=3, mode="3d")

        extreme = modes / "extreme.pdb"
        gmx_projection(gmx, prep.ref_pdb, prep.selected_xtc, eig, vec, extreme, first=1, last=3, mode="extr")
        write_porcupine_instructions(modes, max_mode=3)
        for pc in range(1, 4):
            ep = modes / f"extreme{pc}.pdb"
            if ep.exists():
                plot_porcupine(ep, modes / f"PORCUPINE_PC{pc}",
                               f"{prep.spec.name} — Porcupine PC{pc} (individual PCA)", dpi)

    # Convergence / cosine content over increasing time windows
    banner(f"PCA CONVERGENCE: {prep.spec.name}")
    conv_rows = []
    for end_ns in convergence_endpoints(prep.last_time_ns, convergence_step_ns, prep.first_time_ns):
        tag = f"0_{end_ns:g}ns".replace(".", "p")
        try:
            e_i, v_i, _, _ = gmx_covar(
                gmx, prep.ref_pdb, prep.selected_xtc, conv,
                n_pcs=n_pcs,
                mass_weighted=mass_weighted,
                ref_mode_first=ref_mode_first,
                begin_ns=prep.first_time_ns,
                end_ns=end_ns,
                tag=tag,
            )
            pr_i = conv / f"projection_{tag}.xvg"
            gmx_projection(gmx, prep.ref_pdb, prep.selected_xtc, e_i, v_i, pr_i, first=1, last=ncc, mode="proj")
            cc_i = conv / f"cosine_{tag}.xvg"
            gmx_cosine(gmx, pr_i, cc_i, ncc)
            cx, cy = read_xvg_xy(cc_i)
            row = {"end_ns": end_ns}
            for pcx, ccy in zip(cx, cy):
                row[f"PC{int(round(pcx))}"] = float(ccy)
            conv_rows.append(row)
        except Exception as exc:
            print(f"[convergence][warning] window ending {end_ns} ns failed: {exc}")

    if conv_rows:
        conv_csv = conv / "cosine_convergence.csv"
        all_keys = ["end_ns"] + [f"PC{i}" for i in range(1, ncc + 1)]
        with conv_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader(); w.writerows(conv_rows)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ends = [r["end_ns"] for r in conv_rows]
        for i in range(1, ncc + 1):
            ys = [r.get(f"PC{i}", np.nan) for r in conv_rows]
            ax.plot(ends, ys, marker="o", linewidth=1, label=f"PC{i}")
        ax.set_xlabel("Trajectory end time (ns)")
        ax.set_ylabel("Cosine content")
        ax.set_ylim(0, 1.05)
        ax.set_title("PCA Convergence: Cosine Content vs Trajectory Length")
        ax.legend(ncol=2, fontsize=8)
        savefig(fig, plots / "cosine_convergence", dpi)

    # FEL
    fel_metrics: Dict[str, Dict[str, float]] = {}
    if make_fel:
        banner(f"FREE ENERGY LANDSCAPE: {prep.spec.name}")
        for pair, proj in proj_paths.items():
            files = gmx_sham(gmx, proj, fel, pair, temperature)
            m = guaranteed_fel(files, proj, pair, fel, f"FEL_PC{pair[0]}_PC{pair[1]}", dpi, temperature)
            fel_metrics[f"PC{pair[0]}_PC{pair[1]}"] = m
            plot_xpm_heatmap(files["prob"], fel / f"Probability_PC{pair[0]}_PC{pair[1]}",
                             f"Probability: PC{pair[0]} vs PC{pair[1]}", "Probability", dpi)
    metrics["FEL"] = fel_metrics

    write_single_summary(prep, metrics, root / "analysis_summary.txt", temperature)
    (root / "analysis_manifest.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "metrics": metrics,
        "eig": eig,
        "vec": vec,
        "avg": avg,
        "proj": proj_paths,
        "prepared": prep,
    }


def write_porcupine_instructions(modes_dir: Path, max_mode: int = 3) -> None:
    p = modes_dir / "PORCUPINE_README.txt"
    p.write_text(textwrap.dedent(f"""
        Extreme structures were generated by `gmx anaeig -extr` for PCs 1-{max_mode}.

        The original tutorial.zip also contains modevectors.py and a PyMOL
        porcupine_plot_pymol.pml workflow. Place/copy modevectors.py into this
        directory and adapt the PyMOL object names to the extreme*.pdb files
        generated here.

        Typical PyMOL sequence for one mode:
          load extreme1.pdb
          run modevectors.py
          split_states extreme1
          modevectors extreme1_0001, extreme1_0030, cutoff=0.0, head_length=1, head=0.4, headrgb=(1,0,0), notail=0
    """).strip() + "\n", encoding="utf-8")


def write_single_summary(prep: PreparedSystem, metrics: Dict[str, object], path: Path, temperature: float) -> None:
    lines = [
        f"PCA/FEL analysis summary — {prep.spec.name}",
        f"Script version: {SCRIPT_VERSION}",
        "",
        f"Topology: {prep.spec.tpr}",
        f"Trajectory: {prep.spec.traj}",
        f"Index: {prep.spec.ndx}",
        f"Index group: {prep.spec.group}",
        f"MDAnalysis selection: {prep.spec.selection}",
        f"Atom mode: {prep.spec.atom_mode}",
        f"Resid filter: {prep.spec.resids}",
        f"Selected atoms: {prep.n_atoms}",
        f"Analyzed frames: {prep.n_frames}",
        f"Time range: {prep.first_time_ns:.3f}–{prep.last_time_ns:.3f} ns",
        f"FEL temperature: {temperature:.2f} K",
        "",
        "Variance summary:",
    ]
    for k, v in metrics.items():
        if k.endswith("_variance_pct") or k.startswith("cumulative_"):
            lines.append(f"  {k}: {v:.4f}")
    lines += ["", "Cosine content:"]
    for k, v in metrics.items():
        if k.endswith("_cosine"):
            lines.append(f"  {k}: {v:.5f}")
    lines += ["", "FEL minima (best-effort from GROMACS XPM):"]
    for pair, d in (metrics.get("FEL") or {}).items():
        lines.append(f"  {pair}: {d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Comparison engine
# =============================================================================

def _normalized_atom_identity(sig: Tuple[int, str, str]) -> Tuple[int, str, str]:
    """Canonical atom identity used for cross-system matching.

    Matching is intentionally based ONLY on the chemically relevant identity
    already captured from the original topology selection:
        (resid, resname, atomname)

    String fields are stripped and upper-cased so harmless formatting/case
    differences cannot make otherwise identical selected atoms look different.
    """
    resid, resname, atomname = sig
    return (int(resid), str(resname).strip().upper(), str(atomname).strip().upper())


def _signature_occurrence_keys(signature: Sequence[Tuple[int, str, str]]) -> List[Tuple[int, str, str, int]]:
    """Occurrence-aware canonical keys for deterministic duplicate handling."""
    counts: Dict[Tuple[int, str, str], int] = {}
    out: List[Tuple[int, str, str, int]] = []
    for sig in signature:
        base = _normalized_atom_identity(sig)
        counts[base] = counts.get(base, 0) + 1
        out.append((*base, counts[base]))
    return out


def _subset_prepared_system(prep: PreparedSystem, positions: Sequence[int], tag: str) -> PreparedSystem:
    """Create a PreparedSystem containing only selected positions from an already prepared trajectory."""
    positions = np.asarray(list(positions), dtype=int)
    if positions.size == 0:
        raise ValueError(f"{prep.spec.name}: automatic common-atom matching produced zero atoms.")

    outdir = prep.outdir / tag
    outdir.mkdir(parents=True, exist_ok=True)
    ref_pdb = outdir / "ref_common_atoms.pdb"
    selected_xtc = outdir / "selected_common_atoms.xtc"
    selection_ndx = outdir / "common_atoms.ndx"
    atom_table = outdir / "common_atoms.tsv"

    u = mda.Universe(str(prep.ref_pdb), str(prep.selected_xtc))
    ag = u.atoms[positions]

    u.trajectory[0]
    ag.write(str(ref_pdb))
    with mda.Writer(str(selected_xtc), n_atoms=ag.n_atoms) as W:
        for ts in u.trajectory:
            W.write(ag)

    write_ndx(selection_ndx, "CommonSelection", list(range(1, ag.n_atoms + 1)))

    signature = [prep.signature[int(i)] for i in positions]
    with atom_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["common_order", "source_selected_order", "resid", "resname", "atomname"])
        for j, (src, sig) in enumerate(zip(positions, signature), start=1):
            resid, resname, atomname = sig
            w.writerow([j, int(src) + 1, resid, resname, atomname])

    print(f"[comparison] {prep.spec.name}: common matched selection -> {ag.n_atoms} atoms")
    return PreparedSystem(
        spec=prep.spec,
        outdir=outdir,
        selected_xtc=selected_xtc,
        ref_pdb=ref_pdb,
        selection_ndx=selection_ndx,
        atom_table=atom_table,
        n_atoms=int(ag.n_atoms),
        n_frames=prep.n_frames,
        first_time_ns=prep.first_time_ns,
        last_time_ns=prep.last_time_ns,
        signature=signature,
    )


def harmonize_for_common_pca(a: PreparedSystem, b: PreparedSystem, strict: bool) -> Tuple[PreparedSystem, PreparedSystem]:
    """Return atom-compatible systems using exact identities from the ORIGINAL selections.

    Important:
      * Matching uses the signatures captured directly from each original
        TPR + NDX selection before writing any intermediate PDB/XTC.
      * Identity = resid + resname + atomname (+ occurrence only for a true duplicate).
      * Atom order in either index file is allowed to differ.
      * If both selections contain the same identities, System 2 is reordered
        to System 1 order and ALL atoms are retained.
      * No residue is silently dropped merely because the index ordering differs.
    """
    ka = _signature_occurrence_keys(a.signature)
    kb = _signature_occurrence_keys(b.signature)

    # Fast path: same identities already in the same order.
    if a.n_atoms == b.n_atoms and ka == kb:
        print("[comparison] Atom selections match exactly by resid/resname/atomname and order.")
        return a, b

    # Map each exact occurrence-aware identity to its position in System 2.
    pos_b: Dict[Tuple[int, str, str, int], int] = {k: i for i, k in enumerate(kb)}
    set_a, set_b = set(ka), set(kb)

    ia: List[int] = []
    ib: List[int] = []
    common_keys: List[Tuple[int, str, str, int]] = []
    for i, k in enumerate(ka):
        j = pos_b.get(k)
        if j is not None:
            ia.append(i)
            ib.append(j)
            common_keys.append(k)

    n_common = len(common_keys)
    smaller = min(a.n_atoms, b.n_atoms)
    frac = n_common / smaller if smaller else 0.0

    print("\n" + "-" * 78)
    print("EXACT COMMON-ATOM IDENTITY CHECK")
    print("-" * 78)
    print("Matching key: resid + resname + atomname (occurrence only for duplicate labels)")
    print(f"{a.spec.name}: selected {a.n_atoms}; exact shared {n_common}; unmatched {a.n_atoms - n_common}")
    print(f"{b.spec.name}: selected {b.n_atoms}; exact shared {n_common}; unmatched {b.n_atoms - n_common}")
    print(f"Shared fraction relative to smaller selection: {100.0*frac:.1f}%")

    # Diagnostic file: makes any future mismatch explicit instead of mysterious.
    diag_root = a.outdir.parent / "COMMON_PCA_COMPARISON"
    diag_root.mkdir(parents=True, exist_ok=True)
    diag = diag_root / "COMMON_ATOM_IDENTITY_CHECK.tsv"
    with diag.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["status", "resid", "resname", "atomname", "occurrence",
                    f"{a.spec.name}_selected_order", f"{b.spec.name}_selected_order"])
        pos_a = {k: i for i, k in enumerate(ka)}
        for k in ka:
            if k in set_b:
                w.writerow(["MATCH", k[0], k[1], k[2], k[3],
                            pos_a[k] + 1, pos_b[k] + 1])
            else:
                w.writerow([f"ONLY_{a.spec.name}", k[0], k[1], k[2], k[3],
                            pos_a[k] + 1, ""])
        for k in kb:
            if k not in set_a:
                w.writerow([f"ONLY_{b.spec.name}", k[0], k[1], k[2], k[3],
                            "", pos_b[k] + 1])
    print(f"[comparison] Identity audit: {diag}")

    if n_common < 3 or frac < 0.50:
        only_a = [k for k in ka if k not in set_b]
        only_b = [k for k in kb if k not in set_a]
        sample_a = ", ".join(f"{r}:{rn}:{an}" for r, rn, an, _ in only_a[:12]) or "none"
        sample_b = ", ".join(f"{r}:{rn}:{an}" for r, rn, an, _ in only_b[:12]) or "none"
        raise ValueError(
            "The two ORIGINAL selections do not contain enough identical atoms for common PCA.\n"
            f"Exact shared atoms: {n_common}/{smaller} ({100.0*frac:.1f}%).\n"
            f"Unmatched in {a.spec.name}: {sample_a}\n"
            f"Unmatched in {b.spec.name}: {sample_b}\n"
            f"Full identity audit: {diag}\n"
            "This comparison is stopped rather than silently inventing atom correspondence."
        )

    # Same identity set, different ordering: reorder System 2 only.
    # This is the important case for independently generated NDX groups.
    if n_common == a.n_atoms == b.n_atoms and set_a == set_b:
        print("[comparison] Same exact atoms are present in both selections, but their order differs.")
        print("[comparison] Reordering System 2 to System 1 identity order; NO atoms will be discarded.")
        ca = _subset_prepared_system(a, ia, "00_common_matched")
        cb = _subset_prepared_system(b, ib, "00_common_matched")
    else:
        # Genuine structural difference: retain only exact shared identities.
        print("[comparison] Genuine identity differences exist; retaining only exact shared atoms.")
        ca = _subset_prepared_system(a, ia, "00_common_matched")
        cb = _subset_prepared_system(b, ib, "00_common_matched")

    # Compare canonical identities, not raw formatting.
    if ca.n_atoms != cb.n_atoms:
        raise RuntimeError("Internal common-atom matching error: matched atom counts differ.")
    if _signature_occurrence_keys(ca.signature) != _signature_occurrence_keys(cb.signature):
        raise RuntimeError("Internal common-atom matching error: matched atom identities/order still differ.")

    print(f"[comparison] Common PCA exact matched atoms: {ca.n_atoms}")
    print("-" * 78 + "\n")
    return ca, cb


def compare_signatures(a: PreparedSystem, b: PreparedSystem, strict: bool) -> None:
    """Compatibility check retained for exact comparisons."""
    if a.n_atoms != b.n_atoms:
        raise ValueError(
            "The two PCA selections contain different atom counts after harmonization:\n"
            f"  {a.spec.name}: {a.n_atoms}\n"
            f"  {b.spec.name}: {b.n_atoms}"
        )
    if strict and _signature_occurrence_keys(a.signature) != _signature_occurrence_keys(b.signature):
        raise ValueError("The two PCA selections still have different canonical atom identities after harmonization.")


def histogram_overlap_2d(x1, y1, x2, y2, bins: int = 60) -> float:
    xmin = min(np.min(x1), np.min(x2)); xmax = max(np.max(x1), np.max(x2))
    ymin = min(np.min(y1), np.min(y2)); ymax = max(np.max(y1), np.max(y2))
    if xmax <= xmin or ymax <= ymin:
        return float("nan")
    h1, _, _ = np.histogram2d(x1, y1, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    h2, _, _ = np.histogram2d(x2, y2, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    if h1.sum() == 0 or h2.sum() == 0:
        return float("nan")
    p1 = h1 / h1.sum(); p2 = h2 / h2.sum()
    return float(np.minimum(p1, p2).sum())


def plot_overlay(p1: Path, p2: Path, pair: Tuple[int, int], name1: str, name2: str, outbase: Path, dpi: int) -> Dict[str, float]:
    x1, y1 = read_2d_projection(p1); x2, y2 = read_2d_projection(p2)
    if len(x1) == 0 or len(x2) == 0:
        return {}
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(x1, y1, s=10, alpha=0.45, label=name1)
    ax.scatter(x2, y2, s=10, alpha=0.45, label=name2)
    c1 = np.array([np.mean(x1), np.mean(y1)])
    c2 = np.array([np.mean(x2), np.mean(y2)])
    ax.scatter([c1[0]], [c1[1]], marker="*", s=120, edgecolor="black")
    ax.scatter([c2[0]], [c2[1]], marker="*", s=120, edgecolor="black")
    ax.set_xlabel(f"PC{pair[0]}"); ax.set_ylabel(f"PC{pair[1]}")
    ax.set_title(f"Common PCA Space: {name1} vs {name2} — PC{pair[0]}/PC{pair[1]}")
    ax.legend()
    savefig(fig, outbase, dpi)
    return {
        "histogram_overlap": histogram_overlap_2d(x1, y1, x2, y2),
        "centroid_separation": float(np.linalg.norm(c1 - c2)),
        f"{name1}_centroid_PC{pair[0]}": float(c1[0]),
        f"{name1}_centroid_PC{pair[1]}": float(c1[1]),
        f"{name2}_centroid_PC{pair[0]}": float(c2[0]),
        f"{name2}_centroid_PC{pair[1]}": float(c2[1]),
    }


def inner_product_and_rmsip(gmx: str, ares: Dict[str, object], bres: Dict[str, object], outdir: Path, n_overlap: int, dpi: int) -> Dict[str, object]:
    outxpm = outdir / f"eigenvector_inner_product_first_{n_overlap}.xpm"
    ref = ares["prepared"].ref_pdb
    cmd = [gmx, "anaeig", "-inpr", outxpm, "-first", "1", "-last", str(n_overlap),
           "-v", ares["vec"], "-v2", bres["vec"], "-eig", ares["eig"], "-eig2", bres["eig"], "-s", ref]
    run_cmd(cmd, stdin="0\n0\n", cwd=outdir, label="Eigenvector inner-product matrix")
    metrics: Dict[str, object] = {"inner_product_xpm": str(outxpm)}
    try:
        mat, _, _ = parse_xpm(outxpm)
        # inner-product XPM may include more than requested; take first NxN.
        sub = np.asarray(mat[:n_overlap, :n_overlap], dtype=float)
        rmsip = math.sqrt(float(np.nansum(sub ** 2)) / max(1, n_overlap))
        metrics["RMSIP_first_N"] = rmsip
        metrics["N"] = n_overlap
        fig, ax = plt.subplots(figsize=(6.3, 5.5))
        im = ax.imshow(sub, origin="lower", aspect="equal", cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label="Eigenvector inner product")
        ax.set_xlabel(ares["prepared"].spec.name)
        ax.set_ylabel(bres["prepared"].spec.name)
        ax.set_title(f"Eigenvector Inner Products (first {n_overlap} PCs)\nRMSIP = {rmsip:.3f}")
        savefig(fig, outdir / f"eigenvector_inner_product_first_{n_overlap}", dpi)
    except Exception as exc:
        print(f"[comparison][warning] could not parse inner-product XPM: {exc}")
    return metrics


def run_compare(
    p1: PreparedSystem,
    p2: PreparedSystem,
    *,
    root: Path,
    gmx: str,
    temperature: float,
    n_pcs: int,
    cosine_pcs: int,
    convergence_step_ns: float,
    mass_weighted: bool,
    ref_mode_first: bool,
    dpi: int,
    strict_signature: bool,
    overlap_pcs: int,
) -> None:
    # For comparison, automatically restrict both selections to the exact atoms
    # shared by the two systems when Apo/Holo selections differ in length.
    p1, p2 = harmonize_for_common_pca(p1, p2, strict_signature)
    compare_signatures(p1, p2, strict_signature)

    # All comparison-mode analyses use the same matched degrees of freedom.
    # This makes individual PCA, eigenvector/RMSIP, and the common basis directly comparable.
    ares = run_single_analysis(p1, gmx=gmx, temperature=temperature, n_pcs=n_pcs,
                               cosine_pcs=cosine_pcs, convergence_step_ns=convergence_step_ns,
                               mass_weighted=mass_weighted, ref_mode_first=ref_mode_first, dpi=dpi)
    bres = run_single_analysis(p2, gmx=gmx, temperature=temperature, n_pcs=n_pcs,
                               cosine_pcs=cosine_pcs, convergence_step_ns=convergence_step_ns,
                               mass_weighted=mass_weighted, ref_mode_first=ref_mode_first, dpi=dpi)

    common = root / "COMMON_PCA_COMPARISON"
    common.mkdir(parents=True, exist_ok=True)
    banner("COMMON PCA BASIS FOR TWO-SYSTEM COMPARISON")

    # Concatenate selected trajectories, exactly as the original tutorial workflow.
    concat = common / "concatenated_selected.xtc"
    run_cmd([gmx, "trjcat", "-f", p1.selected_xtc, p2.selected_xtc, "-o", concat, "-cat"], cwd=common, label="Concatenate selected trajectories")

    eig, vec, avg, cov_xpm = gmx_covar(
        gmx, p1.ref_pdb, concat, common,
        n_pcs=n_pcs, mass_weighted=mass_weighted, ref_mode_first=ref_mode_first,
        tag="common",
    )
    plot_scree(eig, common / "common_scree_plot", n_pcs, dpi)
    plot_xpm_heatmap(cov_xpm, common / "common_covariance_matrix", "Common-Basis Covariance Matrix", "Covariance", dpi, cmap="coolwarm")

    metrics_rows = []
    common_fel_metrics: Dict[str, Dict[str, object]] = {p1.spec.name: {}, p2.spec.name: {}}
    common_proj3: Dict[str, Path] = {}
    proj_common: Dict[str, Dict[Tuple[int, int], Path]] = {p1.spec.name: {}, p2.spec.name: {}}

    for prep in (p1, p2):
        for pair in PC_PAIRS:
            if max(pair) > n_pcs:
                continue
            out = common / f"{safe_name(prep.spec.name)}_common_PC{pair[0]}_PC{pair[1]}.xvg"
            gmx_projection(gmx, p1.ref_pdb, prep.selected_xtc, eig, vec, out, first=pair[0], last=pair[1], mode="2d")
            proj_common[prep.spec.name][pair] = out

            # System-specific FEL but in the SAME PCA basis.
            fel_dir = common / f"FEL_{safe_name(prep.spec.name)}_common_basis"
            fel_dir.mkdir(exist_ok=True)
            files = gmx_sham(gmx, out, fel_dir, pair, temperature)
            fm = guaranteed_fel(files, out, pair, fel_dir, f"{safe_name(prep.spec.name)}_FEL_PC{pair[0]}_PC{pair[1]}", dpi, temperature)
            common_fel_metrics[prep.spec.name][f"PC{pair[0]}-PC{pair[1]}"] = fm

        # Explicit numeric PC1/PC2/PC3 projection for a true 3D PCA scatter figure.
        if n_pcs >= 3:
            p3x = common / f"{safe_name(prep.spec.name)}_common_PC1_PC2_PC3.xvg"
            gmx_projection(gmx, p1.ref_pdb, prep.selected_xtc, eig, vec, p3x, first=1, last=3, mode="proj")
            common_proj3[prep.spec.name] = p3x
            plot_pca_3d_from_proj(p3x, common / f"{safe_name(prep.spec.name)}_COMMON_PCA_PC1_PC2_PC3_3D", dpi,
                                  f"{prep.spec.name} — Common-basis PCA 3D")

        # Match the original tutorial capability: 3D projection of each system
        # onto the common PC1/PC2/PC3 basis.
        if n_pcs >= 3:
            out3d = common / f"{safe_name(prep.spec.name)}_common_PC1_PC2_PC3.pdb"
            gmx_projection(gmx, p1.ref_pdb, prep.selected_xtc, eig, vec, out3d, first=1, last=3, mode="3d")

    common_3d_metrics = {}
    if p1.spec.name in common_proj3 and p2.spec.name in common_proj3:
        common_3d_metrics = plot_common_pca_3d_overlay(
            common_proj3[p1.spec.name], common_proj3[p2.spec.name], p1.spec.name, p2.spec.name,
            common / "OVERLAY_PC1_PC2_PC3_3D", dpi)

    # Rigorous Porcupine comparison.
    # (A) Common-basis: SAME eigenvector; compare robust projection amplitude only.
    # (B) Individual PCA: compare mode direction/shape after Kabsch + sign alignment + RMS normalization.
    por_common = common / "PORCUPINE_COMMON_BASIS"
    por_common.mkdir(exist_ok=True)
    common_sampling_metrics: List[Dict[str, object]] = []
    individual_direction_metrics: List[Dict[str, object]] = []

    if n_pcs >= 3:
        # One common-basis extreme structure is sufficient to define the shared mode shape.
        common_extremes: Dict[int, Path] = {}
        for pc in range(1, 4):
            ep = por_common / f"COMMON_extreme_PC{pc}.pdb"
            gmx_projection(gmx, p1.ref_pdb, concat, eig, vec, ep, first=pc, last=pc, mode="extr")
            common_extremes[pc] = ep

            # Use the already generated common-basis 2D projection tables.
            # They contain exactly two numeric columns and NO time column:
            # PC1 -> column 0 of PC1-PC2
            # PC2 -> column 1 of PC1-PC2
            # PC3 -> column 1 of PC1-PC3
            if pc == 1:
                ca = common / f"{safe_name(p1.spec.name)}_common_PC1_PC2.xvg"
                cb = common / f"{safe_name(p2.spec.name)}_common_PC1_PC2.xvg"
                proj_column = 0
            elif pc == 2:
                ca = common / f"{safe_name(p1.spec.name)}_common_PC1_PC2.xvg"
                cb = common / f"{safe_name(p2.spec.name)}_common_PC1_PC2.xvg"
                proj_column = 1
            else:
                ca = common / f"{safe_name(p1.spec.name)}_common_PC1_PC3.xvg"
                cb = common / f"{safe_name(p2.spec.name)}_common_PC1_PC3.xvg"
                proj_column = 1
            try:
                common_sampling_metrics.append(
                    rigorous_common_sampling_plot(
                        ep, ca, cb, proj_column, p1.spec.name, p2.spec.name, pc,
                        por_common / f"COMMON_SAMPLING_{safe_name(p1.spec.name)}_vs_{safe_name(p2.spec.name)}_PC{pc}",
                        dpi))
            except Exception as exc:
                print(f"[porcupine][warning] common sampling PC{pc} failed: {exc}")

            ia = ares["prepared"].outdir / "04_modes" / f"extreme{pc}.pdb"
            ib = bres["prepared"].outdir / "04_modes" / f"extreme{pc}.pdb"
            try:
                individual_direction_metrics.append(
                    rigorous_individual_direction_plot(
                        ia, ib, p1.spec.name, p2.spec.name, pc,
                        por_common / f"INDIVIDUAL_DIRECTION_{safe_name(p1.spec.name)}_vs_{safe_name(p2.spec.name)}_PC{pc}",
                        por_common / f"INDIVIDUAL_DIRECTION_PC{pc}_RESIDUES.csv",
                        dpi))
            except Exception as exc:
                print(f"[porcupine][warning] individual direction PC{pc} failed: {exc}")

        _write_rows_csv(por_common / "PORCUPINE_COMMON_SAMPLING_METRICS.csv", common_sampling_metrics)
        _write_rows_csv(por_common / "PORCUPINE_INDIVIDUAL_MODE_METRICS.csv", individual_direction_metrics)

        por_lines = [
            "Rigorous Porcupine comparison",
            f"Script version: {SCRIPT_VERSION}", "",
            "COMMON-BASIS MODE:",
            "  The eigenvector direction is shared by construction.",
            "  The two panels therefore compare P95-P05 projection sampling amplitude using one shared arrow scale.",
            "  Do NOT interpret these panels as a comparison of different motion directions.", "",
            "INDIVIDUAL-PCA MODE:",
            "  First structures are Kabsch aligned; the arbitrary PCA sign is aligned; displacement fields are RMS normalized.",
            "  These panels compare mode shape/direction independently of absolute amplitude.",
            "  Residue CSV files rank local vector differences.", "",
            "Common sampling metrics:",
            json.dumps(common_sampling_metrics, indent=2, default=str), "",
            "Individual direction metrics:",
            json.dumps(individual_direction_metrics, indent=2, default=str),
        ]
        (por_common / "PORCUPINE_MOTION_SUMMARY.txt").write_text("\n".join(por_lines)+"\n", encoding="utf-8")

    for pair in PC_PAIRS:
        if pair not in proj_common[p1.spec.name] or pair not in proj_common[p2.spec.name]:
            continue
        met = plot_overlay(
            proj_common[p1.spec.name][pair], proj_common[p2.spec.name][pair], pair,
            p1.spec.name, p2.spec.name,
            common / f"OVERLAY_PC{pair[0]}_PC{pair[1]}", dpi,
        )
        row = {"pair": f"PC{pair[0]}-PC{pair[1]}", **met}
        metrics_rows.append(row)

    # Eigenvector comparison of INDIVIDUAL PCA bases. Reproduce the useful
    # tutorial comparisons for the first 2 PCs, first 3 PCs, and requested N.
    overlap_sizes = []
    for n in (2, 3, min(overlap_pcs, n_pcs)):
        if 1 <= n <= n_pcs and n not in overlap_sizes:
            overlap_sizes.append(n)
    ev = {}
    for n in overlap_sizes:
        ev[f"first_{n}_PCs"] = inner_product_and_rmsip(gmx, ares, bres, common, n, dpi)

    csv_path = common / "comparison_metrics.csv"
    if metrics_rows:
        keys = sorted({k for r in metrics_rows for k in r.keys()}, key=lambda x: (x != "pair", x))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(metrics_rows)

    summary = [
        f"PCA/FEL comparison: {p1.spec.name} vs {p2.spec.name}",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "The two systems were projected onto a common PCA basis built from the concatenated selected trajectories.",
        "Therefore, PC coordinates and FELs in COMMON_PCA_COMPARISON are directly comparable.",
        "",
        f"Selected atoms per system: {p1.n_atoms}",
        f"Strict atom signature check: {strict_signature}",
        f"FEL temperature: {temperature:.2f} K",
        "",
        "Conformational-space comparison metrics:",
    ]
    for r in metrics_rows:
        summary.append(f"  {r}")
    summary += ["", "Eigenvector comparison:", f"  {ev}", "", "Common-basis FEL metrics:"]
    for nm, vals in common_fel_metrics.items():
        summary.append(f"  {nm}: {vals}")
    if common_3d_metrics:
        summary += ["", f"3D common-PCA metrics: {common_3d_metrics}"]

    # Automated conservative interpretation (descriptive, not a claim of causation or thermodynamic convergence).
    summary += ["", "AUTOMATED INTERPRETATION:"]
    for r in metrics_rows:
        ov=r.get("histogram_overlap",float('nan'))
        if np.isfinite(ov):
            level = "very low" if ov < 0.10 else ("low" if ov < 0.25 else ("moderate" if ov < 0.50 else "high"))
            summary.append(f"  {r['pair']}: conformational-space overlap is {ov*100:.2f}% ({level}).")
    rms=[]
    for d in ev.values():
        if isinstance(d,dict) and "RMSIP_first_N" in d: rms.append(float(d["RMSIP_first_N"]))
    if rms:
        summary.append(f"  RMSIP range = {min(rms):.3f}-{max(rms):.3f}; this indicates partial/shared collective-motion subspace despite differences in conformational occupancy.")
    summary.append("  Interpretation rule: PCA/FEL differences indicate redistribution of sampled conformations; they do not alone prove increased stability or a causal ligand effect.")
    summary.append("  Common-basis Porcupine compares sampling amplitude along the SAME shared mode; individual-PCA Porcupine compares mode direction/shape after alignment and normalization.")
    (common / "comparison_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (common / "FINAL_INTERPRETATION_SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "system1": asdict(p1.spec),
        "system2": asdict(p2.spec),
        "common_metrics": metrics_rows,
        "common_3d_metrics": common_3d_metrics,
        "common_fel_metrics": common_fel_metrics,
        "eigenvector_metrics": ev,
    }
    # Convert Paths to strings for JSON.
    def default_json(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    (common / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, default=default_json), encoding="utf-8")

    # Self-contained navigable HTML report (images referenced relatively).
    try:
        report = write_comparison_html(
            common, p1.spec.name, p2.spec.name, metrics_rows, ev,
            common_sampling_metrics, individual_direction_metrics)
        print(f"[html] report : {report}")
    except Exception as exc:
        print(f"[html][warning] report generation failed: {exc}")


# =============================================================================
# CLI
# =============================================================================

def parse_resids(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    toks = re.split(r"[\s,;]+", text.strip())
    vals = [int(x) for x in toks if x]
    return vals or None


def add_common_analysis_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gmx", default="gmx", help="GROMACS executable name/path (default: gmx)")
    p.add_argument("--temperature", type=float, help="Simulation temperature in K for gmx sham/FEL. If omitted, interactive mode asks for it.")
    p.add_argument("--n-pcs", type=int, default=10, help="Number of eigenvectors to calculate/use (default: 10)")
    p.add_argument("--cosine-pcs", type=int, default=10, help="Number of PCs for cosine content (default: 10)")
    p.add_argument("--convergence-step-ns", type=float, default=10.0, help="PCA convergence window step in ns; <=0 disables")
    p.add_argument("--no-mass-weight", action="store_true", help="Disable mass-weighted covariance (-mwa)")
    p.add_argument("--covar-reference", choices=["first", "average"], default="first",
                   help="Use first-frame reference (-ref) or average-reference covariance (default: first, matching tutorial)")
    p.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")


def add_selection_args(p: argparse.ArgumentParser, suffix: str = "") -> None:
    s = suffix
    p.add_argument(f"--ndx{s}", type=Path, help="GROMACS .ndx file (required when --group is used)")
    p.add_argument(f"--group{s}", help="Exact GROMACS index group name to analyze")
    p.add_argument(f"--selection{s}", help="MDAnalysis selection string instead of an index group")
    p.add_argument(f"--atoms{s}", choices=["all", "heavy", "backbone", "calpha"],
                   help="Atoms from the requested region to use for PCA. If omitted, interactive mode shows a numbered menu.")
    p.add_argument(f"--resids{s}", help="Optional residue-number filter, e.g. '178 180 208 267'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Flexible GROMACS PCA/FEL analysis for one system or two-system comparison.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single", formatter_class=argparse.RawTextHelpFormatter,
                       help="Analyze one MD system")
    s.add_argument("--name", help="Optional system label. If omitted, derived from the TPR filename.")
    s.add_argument("--tpr", type=Path)
    s.add_argument("--traj", type=Path)
    add_selection_args(s)
    s.add_argument("--start-ns", type=float)
    s.add_argument("--stop-ns", type=float)
    s.add_argument("--frame-stride", type=int, default=1)
    s.add_argument("--output", type=Path)
    add_common_analysis_options(s)

    c = sub.add_parser("compare", formatter_class=argparse.RawTextHelpFormatter,
                       help="Analyze and compare two MD systems")
    c.add_argument("--name1")
    c.add_argument("--tpr1", type=Path)
    c.add_argument("--traj1", type=Path)
    add_selection_args(c, "1")
    c.add_argument("--start-ns1", type=float)
    c.add_argument("--stop-ns1", type=float)
    c.add_argument("--frame-stride1", type=int, default=1)

    c.add_argument("--name2")
    c.add_argument("--tpr2", type=Path)
    c.add_argument("--traj2", type=Path)
    add_selection_args(c, "2")
    c.add_argument("--start-ns2", type=float)
    c.add_argument("--stop-ns2", type=float)
    c.add_argument("--frame-stride2", type=int, default=1)

    c.add_argument("--output", type=Path)
    c.add_argument("--allow-signature-mismatch", action="store_true",
                   help="Allow same-count selections whose resid/resname/atom-name signatures differ. Use only when ordering is known equivalent.")
    c.add_argument("--overlap-pcs", type=int, default=10, help="Number of PCs for eigenvector RMSIP/inner-product comparison")
    add_common_analysis_options(c)
    return parser




def _prompt_path(label: str, suffixes: Tuple[str, ...]) -> Path:
    while True:
        raw = input(f"{label}: ").strip().strip('\"').strip("'")
        if not raw:
            print("  Please enter a file path.")
            continue
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            print(f"  File not found: {p}")
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            print(f"  Expected file extension: {', '.join(suffixes)}")
            continue
        return p


def _prompt_temperature() -> float:
    while True:
        raw = input("Simulation temperature (K): ").strip()
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError
            return val
        except ValueError:
            print("  Enter a positive number, e.g. 300.")


def _prompt_atom_mode() -> str:
    modes = [(1, "all"), (2, "heavy"), (3, "backbone"), (4, "calpha")]
    print("\nChoose atoms from the selected index group:")
    for i, name in modes:
        print(f"  {i:>2}  {name}")
    while True:
        raw = input("Atom mode number: ").strip()
        try:
            n = int(raw)
        except ValueError:
            print("  Enter one of the displayed numbers.")
            continue
        for i, name in modes:
            if n == i:
                return name
        print("  Enter one of the displayed numbers.")


def _prompt_group_from_ndx(ndx: Path, label: str) -> str:
    groups = parse_ndx(ndx)
    names = list(groups.keys())
    if not names:
        raise ValueError(f"No groups found in index file: {ndx}")
    print(f"\nIndex groups for {label}: {ndx}")
    for i, name in enumerate(names):
        print(f"  {i:>3}  {name:<30} ({len(groups[name])} atoms)")
    while True:
        raw = input("Enter the index group NUMBER to analyze: ").strip()
        try:
            n = int(raw)
        except ValueError:
            print("  Enter one of the displayed numbers.")
            continue
        if 0 <= n < len(names):
            chosen = names[n]
            print(f"Selected group: {n} -> {chosen}")
            return chosen
        print("  Enter one of the displayed numbers.")


def _interactive_fill(args) -> None:
    """Fill missing core inputs interactively; no analysis group is ever assumed."""
    print("\n=== Interactive input ===")
    if args.mode == "single":
        if args.tpr is None:
            args.tpr = _prompt_path("TPR file (.tpr)", (".tpr",))
        if args.traj is None:
            args.traj = _prompt_path("Trajectory file (.xtc/.trr)", (".xtc", ".trr"))
        if args.ndx is None and not args.selection:
            args.ndx = _prompt_path("Index file (.ndx)", (".ndx",))
        if not args.group and not args.selection:
            args.group = _prompt_group_from_ndx(args.ndx, "system")
        if args.atoms is None:
            args.atoms = _prompt_atom_mode()
        if args.temperature is None:
            args.temperature = _prompt_temperature()
        if not args.name:
            args.name = args.tpr.stem
        if args.output is None:
            tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.group or "selection")
            args.output = Path(f"PCA_FEL_{args.name}_{tag}")
        return

    for i in (1, 2):
        if getattr(args, f"tpr{i}") is None:
            setattr(args, f"tpr{i}", _prompt_path(f"System {i} TPR file (.tpr)", (".tpr",)))
        if getattr(args, f"traj{i}") is None:
            setattr(args, f"traj{i}", _prompt_path(f"System {i} trajectory (.xtc/.trr)", (".xtc", ".trr")))
        if getattr(args, f"ndx{i}") is None and not getattr(args, f"selection{i}"):
            setattr(args, f"ndx{i}", _prompt_path(f"System {i} index file (.ndx)", (".ndx",)))
        if not getattr(args, f"group{i}") and not getattr(args, f"selection{i}"):
            setattr(args, f"group{i}", _prompt_group_from_ndx(getattr(args, f"ndx{i}"), f"system {i}"))
        if getattr(args, f"atoms{i}") is None:
            setattr(args, f"atoms{i}", _prompt_atom_mode())
        if not getattr(args, f"name{i}"):
            setattr(args, f"name{i}", getattr(args, f"tpr{i}").stem)
    if args.temperature is None:
        args.temperature = _prompt_temperature()
    if args.output is None:
        args.output = Path(f"Comparison_{args.name1}_vs_{args.name2}")

def make_spec_single(args) -> SystemSpec:
    return SystemSpec(
        name=args.name,
        tpr=ensure_file(args.tpr, "TPR"),
        traj=ensure_file(args.traj, "trajectory"),
        ndx=ensure_file(args.ndx, "index") if args.ndx else None,
        group=args.group,
        selection=args.selection,
        atom_mode=args.atoms,
        resids=parse_resids(args.resids),
        start_ns=args.start_ns,
        stop_ns=args.stop_ns,
        frame_stride=args.frame_stride,
    )


def make_spec_compare(args, i: int) -> SystemSpec:
    return SystemSpec(
        name=getattr(args, f"name{i}"),
        tpr=ensure_file(getattr(args, f"tpr{i}"), f"TPR{i}"),
        traj=ensure_file(getattr(args, f"traj{i}"), f"trajectory{i}"),
        ndx=ensure_file(getattr(args, f"ndx{i}"), f"index{i}") if getattr(args, f"ndx{i}") else None,
        group=getattr(args, f"group{i}"),
        selection=getattr(args, f"selection{i}"),
        atom_mode=getattr(args, f"atoms{i}"),
        resids=parse_resids(getattr(args, f"resids{i}")),
        start_ns=getattr(args, f"start_ns{i}"),
        stop_ns=getattr(args, f"stop_ns{i}"),
        frame_stride=getattr(args, f"frame_stride{i}"),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _interactive_fill(args)
    gmx = find_gmx(args.gmx)
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    banner("PCA / FEL ANALYZER")
    print(f"Script version : {SCRIPT_VERSION}")
    print(f"Mode           : {args.mode}")
    print(f"GROMACS        : {gmx}")
    print(f"Output         : {root}")
    print(f"Temperature    : {args.temperature} K")
    print(f"PCA PCs        : {args.n_pcs}")
    print(f"Cosine PCs     : {args.cosine_pcs}")

    if args.mode == "single":
        spec = make_spec_single(args)
        prep = prepare_system(spec, root)
        run_single_analysis(
            prep,
            gmx=gmx,
            temperature=args.temperature,
            n_pcs=args.n_pcs,
            cosine_pcs=args.cosine_pcs,
            convergence_step_ns=args.convergence_step_ns,
            mass_weighted=not args.no_mass_weight,
            ref_mode_first=(args.covar_reference == "first"),
            dpi=args.dpi,
        )
        banner("FINISHED")
        print(f"Results: {prep.outdir}")
        return 0

    spec1 = make_spec_compare(args, 1)
    spec2 = make_spec_compare(args, 2)
    p1 = prepare_system(spec1, root)
    p2 = prepare_system(spec2, root)
    run_compare(
        p1, p2,
        root=root,
        gmx=gmx,
        temperature=args.temperature,
        n_pcs=args.n_pcs,
        cosine_pcs=args.cosine_pcs,
        convergence_step_ns=args.convergence_step_ns,
        mass_weighted=not args.no_mass_weight,
        ref_mode_first=(args.covar_reference == "first"),
        dpi=args.dpi,
        strict_signature=not args.allow_signature_mismatch,
        overlap_pcs=args.overlap_pcs,
    )
    banner("FINISHED")
    print(f"Results: {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print("\n" + "!" * 78, file=sys.stderr)
        print("ERROR", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        raise SystemExit(1)
