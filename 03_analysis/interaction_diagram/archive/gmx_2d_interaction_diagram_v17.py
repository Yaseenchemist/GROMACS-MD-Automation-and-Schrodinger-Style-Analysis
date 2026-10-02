#!/usr/bin/env python3
"""
2D protein-ligand interaction diagram from GROMACS trajectory — Schrödinger-style.

v5 — Reliable detection of ALL interaction types
-------------------------------------------------
ProLIF handles: HBond · Hydrophobic · PiStacking · PiCation · Halogen
Custom geometry handles (100% reliable, version-independent):
  • WaterBridge  — water O within 3.5 Å of BOTH ligand H-bond atom AND protein H-bond atom
  • Ionic        — ASP/GLU oxygens OR ARG/LYS/HIS+ nitrogens within 4.5 Å of ligand charged atoms
  • Metal        — ZN/MG/CA/… within 2.8 Å of ligand N/O/S atoms

Requires: MDAnalysis, prolif, rdkit, matplotlib, numpy, scipy

Example
-------
python3 gmx_2d_interaction_diagram.py \\
    --topol MD.tpr \\
    --traj MD_center.xtc \\
    --ligand-selection "resname LIG" \\
    --protein-selection "protein" \\
    --metal-selection "resname ZN ZN2 ZNB MG MG2 CA MN FE FE2 CU NI CO" \\
    --ligand-ref LIG.mol2 \\
    --output-prefix results/ligand_map \\
    --stride 10
"""

from __future__ import annotations

SCRIPT_VERSION = "2026-07-06-v17-ALL-UNITS-NM-TO-ANG"

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Patch
import numpy as np
from scipy.spatial.distance import cdist

import MDAnalysis as mda
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor


# =============================================================================
# INTERACTION STYLES
# =============================================================================

INTERACTION_STYLES: Dict[str, dict] = {
    "HydrogenBond": {
        "color": "#8e44ad",
        "linestyle": (0, (5, 3)),
        "linewidth": 2.0,
        "label_color": "#8e44ad",
        "anchor_mode": "atom",
        "ligand_dot": False,
        "legend": "H-Bond",
    },
    "WaterBridge": {
        "color": "#3477b5",
        "linestyle": (0, (3, 1, 1, 1)),
        "linewidth": 1.9,
        "label_color": "#3477b5",
        "anchor_mode": "atom",
        "ligand_dot": False,
        "legend": "Water Bridge",
    },
    "Hydrophobic": {
        "color": "#27ae60",
        "linestyle": "solid",
        "linewidth": 2.0,
        "label_color": "#27ae60",
        "anchor_mode": "ring_or_atom",
        "ligand_dot": True,
        "legend": "Hydrophobic",
    },
    "PiStacking": {
        "color": "#27ae60",          # Same green as Hydrophobic (merged)
        "linestyle": "solid",
        "linewidth": 2.0,
        "label_color": "#27ae60",
        "anchor_mode": "ring",
        "ligand_dot": True,
        "legend": "Hydrophobic",     # Will never show separately
    },
    "PiCation": {
        "color": "#27ae60",          # Same green as Hydrophobic (merged)
        "linestyle": "solid",
        "linewidth": 2.0,
        "label_color": "#27ae60",
        "anchor_mode": "ring_or_atom",
        "ligand_dot": True,
        "legend": "Hydrophobic",     # Will never show separately
    },
    "Ionic": {
        "color": "#e74c3c",          # RED
        "linestyle": (0, (8, 3)),
        "linewidth": 2.4,
        "label_color": "#c0392b",
        "anchor_mode": "atom",
        "ligand_dot": True,
        "legend": "Ionic / Salt Bridge",
    },
    "Halogen": {
        "color": "#1abc9c",
        "linestyle": (0, (5, 2)),
        "linewidth": 2.0,
        "label_color": "#1abc9c",
        "anchor_mode": "atom",
        "ligand_dot": False,
        "legend": "Halogen Bond",
    },
    "Metal": {
        "color": "#f39c12",          # GOLD
        "linestyle": "solid",
        "linewidth": 2.6,
        "label_color": "#d68910",
        "anchor_mode": "atom",
        "ligand_dot": True,
        "legend": "Metal Coordination",
    },
    "VdWContact": {
        "color": "#95a5a6",
        "linestyle": (0, (1, 2)),
        "linewidth": 1.2,
        "label_color": "#7f8c8d",
        "anchor_mode": "ring_or_atom",
        "ligand_dot": False,
        "legend": "vdW Contact",
    },
}

RESIDUE_NODE_COLORS = {
    "acidic":      ("#e8873a", "#fff0e6"),
    "basic":       ("#8e7dff", "#f0eeff"),
    "polar":       ("#3ec6e0", "#ebfcff"),
    "hydrophobic": ("#a8d832", "#f7ffe1"),
    "aromatic":    ("#a8d832", "#f7ffe1"),
    "metal":       ("#d4ac0d", "#fff6d8"),
    "water":       ("#bdc3c7", "#f8f9fa"),
    "other":       ("#aab0b8", "#f5f6fa"),
}

ACIDIC      = {"ASP", "GLU"}
BASIC       = {"ARG", "LYS"}
POLAR       = {"ASN", "GLN", "SER", "THR", "CYS", "HIS", "HIE", "HID",
               "HIP", "HSE", "HSD", "HSP", "TYR"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "GLY"}
AROMATIC    = {"PHE", "TRP", "TYR", "HIS", "HIE", "HID", "HIP",
               "HSE", "HSD", "HSP"}
METAL_RES   = {"ZN", "ZN2", "ZNB", "MG", "MG2", "MN", "FE", "FE2", "CU",
               "NI", "CO", "CA", "NA", "K", "CD", "PT"}
WATER_NAMES = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE", "SPC/E"}

# ProLIF name → canonical group (only for interactions ProLIF handles)
PROLIF_TO_GROUP: Dict[str, str] = {
    "HBDonor":    "HydrogenBond",
    "HBAcceptor": "HydrogenBond",
    "Hydrophobic": "Hydrophobic",
    "FaceToFace":  "Hydrophobic",   # π–π merged into Hydrophobic
    "EdgeToFace":  "Hydrophobic",
    "PiStacking":  "Hydrophobic",
    "PiCation":    "Hydrophobic",   # π–Cation also merged
    "CationPi":    "Hydrophobic",
    "XBDonor":     "Halogen",
    "XBAcceptor":  "Halogen",
    "VdWContact":  "VdWContact",
    # keep legacy names for safety
    "Anionic":      "Ionic",
    "Cationic":     "Ionic",
    "MetalDonor":   "Metal",
    "MetalAcceptor":"Metal",
    "WaterBridge":  "WaterBridge",
}

# Only ask ProLIF for interactions we fully trust it to detect
_PROLIF_CANDIDATES = [
    "Hydrophobic",
    "HBDonor",
    "HBAcceptor",
    "PiStacking",
    "FaceToFace",
    "EdgeToFace",
    "CationPi",
    "PiCation",
    "XBAcceptor",
    "XBDonor",
    # We still ask but custom will supplement / replace
    "Anionic",
    "Cationic",
    "MetalDonor",
    "MetalAcceptor",
    "WaterBridge",
]

# ── Protein charged group atom names ─────────────────────────────────────────
# Residues with negative formal charge
PROT_NEG_ATOMS: Dict[str, set] = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "CYM": {"SG"},
}
# Residues with positive formal charge
PROT_POS_ATOMS: Dict[str, set] = {
    "ARG": {"NH1", "NH2", "NE"},
    "LYS": {"NZ"},
    "HIP": {"ND1", "NE2"},
    "HSP": {"ND1", "NE2"},
}
# H-bond capable elements
HBOND_EL = {"N", "O", "S", "F"}
# Metal coordination donor elements in ligand
COORD_EL  = {"N", "O", "S"}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class Anchor:
    kind: str
    atom_indices: Tuple[int, ...]
    ring_index: Optional[int]
    coord: np.ndarray
    key: str


@dataclass
class InteractionOccurrence:
    frame: int
    residue_key: str
    residue_name: str
    residue_number: int
    residue_chain: str
    group: str
    prolif_name: str
    percentage_key: str
    anchor: Anchor
    meta: dict
    water_residues: Tuple[str, ...] = ()


@dataclass
class AggregatedInteraction:
    residue_key: str
    residue_name: str
    residue_number: int
    residue_chain: str
    group: str
    prolif_names: List[str]
    percentage: float
    frame_count: int
    anchor: Anchor
    water_order: int = 0
    water_residues: Tuple[str, ...] = ()
    metadata_examples: List[dict] = field(default_factory=list)


@dataclass
class ResidueNode:
    key: str
    name: str
    number: int
    chain: str
    category: str
    radius: float
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0


# =============================================================================
# Helpers
# =============================================================================

def residue_category(resname: str) -> str:
    r = resname.upper()
    if r in WATER_NAMES:  return "water"
    if r in METAL_RES:    return "metal"
    if r in ACIDIC:       return "acidic"
    if r in BASIC:        return "basic"
    if r in AROMATIC:     return "aromatic"
    if r in POLAR:        return "polar"
    if r in HYDROPHOBIC:  return "hydrophobic"
    return "other"

def pretty_residue_label(name, number, chain):
    # chain label removed — only residue name + number shown inside circles
    return f"{name.upper()}\n{number}"

def make_residue_key(name, number, chain):
    return f"{chain}:{name.upper()}:{number}"

def water_residue_to_key(resid):
    return make_residue_key(resid.name, resid.number, resid.chain)

def angle_of(vec):
    return math.atan2(float(vec[1]), float(vec[0]))

def midpoint(a, b):
    return (a + b) / 2.0

def _atom_element(atom) -> str:
    """Get element symbol from MDAnalysis atom, robust to missing element info."""
    el = ""
    try:
        el = atom.element.upper().strip()
    except Exception:
        pass
    if not el or el == "X":
        el = atom.name.strip()[0].upper()
    return el


# =============================================================================
# ProLIF fingerprint builder
# =============================================================================

def _valid_prolif_names() -> List[str]:
    valid = []
    for name in _PROLIF_CANDIDATES:
        try:
            plf.Fingerprint([name])
            valid.append(name)
        except Exception:
            pass
    return valid


def build_fingerprint(water_ag, include_vdw: bool,
                      water_order: int) -> plf.Fingerprint:
    interactions = _valid_prolif_names()
    print(f"[info] ProLIF {plf.__version__} — active interactions: {interactions}")

    if include_vdw:
        try:
            plf.Fingerprint(["VdWContact"])
            if "VdWContact" not in interactions:
                interactions.append("VdWContact")
        except Exception:
            pass

    use_water = water_ag is not None and water_ag.n_atoms > 0
    if not use_water and "WaterBridge" in interactions:
        interactions.remove("WaterBridge")

    if use_water and "WaterBridge" in interactions:
        for kwargs in [
            {"water": water_ag},
            {"parameters": {"WaterBridge": {"water": water_ag,
                                             "order": water_order}}},
        ]:
            try:
                fp = plf.Fingerprint(interactions, **kwargs)
                print("[info] ProLIF WaterBridge enabled")
                return fp
            except TypeError:
                continue
        interactions = [x for x in interactions if x != "WaterBridge"]

    return plf.Fingerprint(interactions)


# =============================================================================
# Molecule preparation
# =============================================================================

def load_reference_mol(path: Optional[str]) -> Optional[Chem.Mol]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warning] Ref file not found: {path}")
        return None
    suffix = p.suffix.lower()
    mol = None
    if suffix in {".sdf", ".mol"}:
        suppl = Chem.SDMolSupplier(str(p), removeHs=False)
        mol = suppl[0] if suppl and len(suppl) else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(p), removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(p), removeHs=False)
    elif suffix == ".smi":
        smi = p.read_text().strip().splitlines()[0].split()[0]
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = Chem.AddHs(mol); AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    if mol is None:
        print(f"[warning] Cannot parse ref mol: {path}")
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def build_depiction_mol(ligand_ag,
                        ligand_ref: Optional[str] = None
                        ) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Return (no-H RDKit mol with 2D coords, idx_map: ag_atom_idx→noH_mol_idx)."""
    mol = None
    try:
        mol = ligand_ag.convert_to("RDKIT")
    except Exception as e:
        print(f"[warning] convert_to(RDKIT): {e}")
    if mol is None:
        try:
            mol = Chem.MolFromPDBBlock(ligand_ag.write("pdb"),
                                        removeHs=False, sanitize=False)
        except Exception as e:
            print(f"[warning] PDB fallback: {e}")
    if mol is None:
        raise RuntimeError("Cannot convert ligand to RDKit mol.")

    if ligand_ref:
        ref = load_reference_mol(ligand_ref)
        if ref is not None:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
                print("[info] Bond orders from ref applied.")
            except Exception as exc:
                print(f"[warning] Bond order transfer: {exc}")

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    # Build idx_map BEFORE removing H
    idx_map: Dict[int, int] = {}
    nh = 0
    for old in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(old).GetAtomicNum() > 1:
            idx_map[old] = nh
            nh += 1

    try:
        mol = Chem.RemoveAllHs(mol)
    except Exception:
        mol = Chem.RemoveHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    rdDepictor.SetPreferCoordGen(True)
    AllChem.Compute2DCoords(mol, clearConfs=True)
    return mol, idx_map


# =============================================================================
# Ring / coordinate utilities
# =============================================================================

def get_2d_atom_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    arr  = np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y]
                     for i in range(mol.GetNumAtoms())], dtype=float)
    arr -= arr.mean(axis=0)
    return arr

def get_aromatic_rings(mol):
    return [tuple(r) for r in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]

def ring_centers(rings, coords):
    return {i: coords[list(r)].mean(axis=0) for i, r in enumerate(rings)}

def choose_ring_for_atoms(atom_indices, rings):
    aset = set(atom_indices); best, bs = None, 0
    for i, ring in enumerate(rings):
        s = len(aset & set(ring))
        if s > bs:
            best, bs = i, s
    return best

def heavy_atom_only(mol, atom_indices):
    n = mol.GetNumAtoms()
    h = [i for i in atom_indices
         if 0 <= i < n and mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if h: return tuple(h)
    v = [i for i in atom_indices if 0 <= i < n]
    return tuple(v) if v else (0,)

def atom_anchor_coord(atom_indices, coords):
    return coords[list(atom_indices)].mean(axis=0)

def make_anchor(mol, coords, rings, centers, atom_indices, mode) -> Anchor:
    atom_indices = heavy_atom_only(mol, atom_indices)
    ring_idx = choose_ring_for_atoms(atom_indices, rings)
    if mode == "ring":
        if ring_idx is None:
            c = atom_anchor_coord(atom_indices, coords)
            return Anchor("atom", tuple(atom_indices), None, c,
                          f"atom:{','.join(map(str, atom_indices))}")
        return Anchor("ring", tuple(atom_indices), ring_idx,
                      centers[ring_idx], f"ring:{ring_idx}")
    if mode == "ring_or_atom" and ring_idx is not None:
        return Anchor("ring", tuple(atom_indices), ring_idx,
                      centers[ring_idx], f"ring:{ring_idx}")
    c = atom_anchor_coord(atom_indices, coords)
    return Anchor("atom", tuple(atom_indices), None, c,
                  f"atom:{','.join(map(str, atom_indices))}")

def ligand_bbox(coords):
    return coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max()


# =============================================================================
# Parse ProLIF output
# =============================================================================

def _safe_indices(meta, idx_map, n_atoms):
    try:
        raw = (meta.get("indices") or {}).get("ligand") or meta.get("ligand_indices")
        if raw is None: return None
        raw_ints = [int(i) for i in raw]
        mapped = [idx_map[i] for i in raw_ints if i in idx_map]
        valid  = [i for i in (mapped or raw_ints) if 0 <= i < n_atoms]
        return tuple(valid) if valid else None
    except Exception:
        return None

def prolif_meta_iter(frame_ifp):
    for pair, idict in frame_ifp.items():
        try:
            lig_res, prot_res = pair
        except (ValueError, TypeError):
            continue
        for name, metas in idict.items():
            if not isinstance(metas, (list, tuple)): metas = [metas]
            for meta in metas:
                if isinstance(meta, dict):
                    yield lig_res, prot_res, name, meta

def parse_prolif_occurrences(fp, n_frames, mol, coords, rings, centers,
                              idx_map) -> List[InteractionOccurrence]:
    occurrences = []
    n_atoms   = mol.GetNumAtoms()
    ifp_items = enumerate(fp.ifp) if isinstance(fp.ifp, list) else fp.ifp.items()
    raw_counts: Dict[str, int] = defaultdict(int)

    for frame_idx, frame_ifp in ifp_items:
        seen = set()
        for lig_res, prot_res, prolif_name, meta in prolif_meta_iter(frame_ifp):
            raw_counts[prolif_name] += 1
            group = PROLIF_TO_GROUP.get(prolif_name)
            if not group:
                if raw_counts[prolif_name] == 1:
                    print(f"[debug] Unmapped ProLIF: {prolif_name!r}")
                continue
            atom_indices = _safe_indices(meta, idx_map, n_atoms) or (0,)
            style  = INTERACTION_STYLES[group]
            anchor = make_anchor(mol, coords, rings, centers,
                                 atom_indices, style["anchor_mode"])
            try:
                res_name   = str(prot_res.name).upper()
                res_number = int(prot_res.number)
                res_chain  = str(prot_res.chain) if prot_res.chain else "?"
            except AttributeError:
                continue
            res_key = make_residue_key(res_name, res_number, res_chain)
            if prolif_name == "WaterBridge":
                wrs = meta.get("water_residues", [])
                w_keys = tuple(water_residue_to_key(wr) for wr in wrs)
                pct_key = f"{res_key}|{group}|{anchor.key}|order:{meta.get('order',1)}"
            else:
                w_keys = (); pct_key = f"{res_key}|{group}|{anchor.key}"
            if (frame_idx, pct_key) in seen: continue
            seen.add((frame_idx, pct_key))
            occurrences.append(InteractionOccurrence(
                frame=int(frame_idx), residue_key=res_key,
                residue_name=res_name, residue_number=res_number,
                residue_chain=res_chain, group=group,
                prolif_name=prolif_name, percentage_key=pct_key,
                anchor=anchor, meta=meta, water_residues=w_keys,
            ))

    if raw_counts:
        print("[info] ProLIF raw counts:")
        for k, v in sorted(raw_counts.items(), key=lambda x: -x[1]):
            print(f"       {k:28s} → {PROLIF_TO_GROUP.get(k,'UNMAPPED'):22s}: {v}")
    return occurrences


# =============================================================================
# CUSTOM GEOMETRIC DETECTORS
# =============================================================================

def _best_lig_anchor(lig_mol_indices_close: List[int],
                     distances_row: np.ndarray,
                     coords_2d: np.ndarray,
                     mol_noH: Chem.Mol) -> Tuple[int, np.ndarray]:
    """Return (mol_noH_idx, 2d_coord) of the closest/best ligand anchor atom."""
    if not lig_mol_indices_close:
        return 0, coords_2d[0]
    best = lig_mol_indices_close[int(np.argmin([distances_row[i]
                                                for i in range(len(lig_mol_indices_close))]))]
    best = min(best, mol_noH.GetNumAtoms() - 1)
    return best, coords_2d[best]


# ── 1. Water Bridge ───────────────────────────────────────────────────────────

def detect_water_bridges(universe, ligand_ag, protein_ag, water_ag,
                          traj_slice, coords_2d, mol_noH, idx_map,
                          cutoff: float = 3.5) -> List[InteractionOccurrence]:
    """
    A water bridge exists when a WATER OXYGEN is within `cutoff` Å of
    BOTH a ligand H-bond atom (N/O/S/F) AND a protein H-bond atom (N/O/S).
    Uses scipy.spatial.distance.cdist for efficiency.
    """
    print("[info] Running custom Water Bridge detector …")

    # ── ligand: indices of H-bond-capable atoms (in ligand_ag order)
    lig_hb_idx = [i for i, a in enumerate(ligand_ag.atoms)
                  if _atom_element(a) in HBOND_EL]
    if not lig_hb_idx:
        print("[warning] No ligand H-bond atoms found; WaterBridge skipped")
        return []

    # ── water: O atoms only
    wat_O_idx = [i for i, a in enumerate(water_ag.atoms)
                 if _atom_element(a) == "O"
                 or a.name.upper() in {"OW", "O", "OH2", "OWT"}]
    if not wat_O_idx:
        print("[warning] No water O atoms found; WaterBridge skipped")
        return []

    # ── protein: H-bond-capable atoms, with residue metadata
    prot_hb_idx = [i for i, a in enumerate(protein_ag.atoms)
                   if _atom_element(a) in HBOND_EL]
    if not prot_hb_idx:
        print("[warning] No protein H-bond atoms found; WaterBridge skipped")
        return []

    lig_hb_arr  = np.array(lig_hb_idx,  dtype=int)
    wat_O_arr   = np.array(wat_O_idx,   dtype=int)
    prot_hb_arr = np.array(prot_hb_idx, dtype=int)
    n_mol = mol_noH.GetNumAtoms()

    occurrences: List[InteractionOccurrence] = []

    for ts in traj_slice:
        frame_idx = ts.frame
        lig_pos  = ligand_ag.positions[lig_hb_arr]   # (nLigHB, 3)
        wat_pos  = water_ag.positions[wat_O_arr]      # (nWatO, 3)
        prot_pos = protein_ag.positions[prot_hb_arr]  # (nProtHB, 3)

        d_wl = cdist(wat_pos, lig_pos)    # (nWatO, nLigHB)
        d_wp = cdist(wat_pos, prot_pos)   # (nWatO, nProtHB)

        close_lig  = d_wl  <= cutoff   # bool (nWatO, nLigHB)
        close_prot = d_wp  <= cutoff   # bool (nWatO, nProtHB)
        bridging   = np.where(close_lig.any(axis=1) & close_prot.any(axis=1))[0]

        seen_frame = set()
        for bw in bridging:
            # Which ligand atoms does this water see?
            lig_close_local = [int(lig_hb_arr[j])
                               for j in np.where(close_lig[bw])[0]]
            # Which protein atoms?
            prot_close_local = [int(prot_hb_arr[j])
                                for j in np.where(close_prot[bw])[0]]

            # Best ligand anchor (closest)
            lig_dists = [d_wl[bw, j] for j in np.where(close_lig[bw])[0]]
            best_lig_ag = lig_close_local[int(np.argmin(lig_dists))]
            lig_mol_idx = idx_map.get(best_lig_ag, 0)
            lig_mol_idx = min(lig_mol_idx, n_mol - 1)
            anchor_coord = coords_2d[lig_mol_idx]

            for pc in prot_close_local:
                pa = protein_ag.atoms[pc]
                res_name   = pa.resname.upper()
                res_number = int(pa.resid)
                res_chain  = pa.segid.strip() or "?"
                res_key    = make_residue_key(res_name, res_number, res_chain)
                pct_key    = f"{res_key}|WaterBridge|atom:{lig_mol_idx}"

                if (frame_idx, pct_key) in seen_frame:
                    continue
                seen_frame.add((frame_idx, pct_key))

                occurrences.append(InteractionOccurrence(
                    frame=int(frame_idx),
                    residue_key=res_key, residue_name=res_name,
                    residue_number=res_number, residue_chain=res_chain,
                    group="WaterBridge", prolif_name="WaterBridge",
                    percentage_key=pct_key,
                    anchor=Anchor("atom", (lig_mol_idx,), None,
                                  anchor_coord, f"atom:{lig_mol_idx}"),
                    meta={"order": 1}, water_residues=(),
                ))

    print(f"[info] WaterBridge custom occurrences: {len(occurrences)}")
    return occurrences


# ── 2. Ionic interactions ─────────────────────────────────────────────────────

def _get_ligand_charged_atoms(mol_noH: Chem.Mol, ligand_ag,
                               idx_map: Dict[int, int]
                               ) -> Tuple[List[int], List[int]]:
    """
    Return (neg_ag_indices, pos_ag_indices) — indices in ligand_ag.atoms.

    Priority:
    1. RDKit formal charges on the no-H mol
    2. Partial charges from MDAnalysis topology if available
    3. Fallback: all O/S atoms as negative, all N atoms as positive
    """
    neg_mol, pos_mol = [], []

    # Try formal charges from RDKit
    for atom in mol_noH.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc < 0:  neg_mol.append(atom.GetIdx())
        elif fc > 0: pos_mol.append(atom.GetIdx())

    # Invert idx_map: noH_mol_idx → list of ag_idx
    inv_map: Dict[int, int] = {v: k for k, v in idx_map.items()}

    if neg_mol or pos_mol:
        neg_ag = [inv_map[i] for i in neg_mol if i in inv_map]
        pos_ag = [inv_map[i] for i in pos_mol if i in inv_map]
        if neg_ag or pos_ag:
            return neg_ag, pos_ag

    # Fallback: try partial charges from topology
    try:
        charges = np.array([a.charge for a in ligand_ag.atoms])
        neg_ag  = [i for i, c in enumerate(charges) if c < -0.3]
        pos_ag  = [i for i, c in enumerate(charges) if c > +0.3]
        if neg_ag or pos_ag:
            return neg_ag, pos_ag
    except Exception:
        pass

    # Last resort: element-based heuristic
    neg_ag, pos_ag = [], []
    for i, a in enumerate(ligand_ag.atoms):
        el = _atom_element(a)
        if el in {"O", "S"}:
            neg_ag.append(i)
        elif el == "N":
            pos_ag.append(i)
    return neg_ag, pos_ag


def detect_ionic(universe, ligand_ag, protein_ag, traj_slice,
                  coords_2d, mol_noH, idx_map,
                  cutoff: float = 4.5) -> List[InteractionOccurrence]:
    """
    Detect ionic / salt-bridge interactions geometrically.

    Protein negative: ASP(OD1,OD2) / GLU(OE1,OE2) → pair with ligand POSITIVE atoms
    Protein positive: ARG(NH1,NH2,NE) / LYS(NZ) / HIP(ND1,NE2) → pair with ligand NEGATIVE atoms
    """
    print("[info] Running custom Ionic detector …")

    # ── Collect protein charged atom indices ─────────────────────────────────
    prot_neg_idx, prot_pos_idx = [], []
    for i, a in enumerate(protein_ag.atoms):
        rn = a.resname.upper()
        nm = a.name.upper()
        if rn in PROT_NEG_ATOMS and nm in PROT_NEG_ATOMS[rn]:
            prot_neg_idx.append(i)
        if rn in PROT_POS_ATOMS and nm in PROT_POS_ATOMS[rn]:
            prot_pos_idx.append(i)

    if not prot_neg_idx and not prot_pos_idx:
        print("[warning] No protein charged atoms found (ASP/GLU/ARG/LYS); Ionic skipped")
        return []

    print(f"[info] Protein negative atoms: {len(prot_neg_idx)} "
          f"(ASP/GLU)   positive: {len(prot_pos_idx)} (ARG/LYS/HIP)")

    # ── Ligand charged atoms ──────────────────────────────────────────────────
    lig_neg_ag, lig_pos_ag = _get_ligand_charged_atoms(mol_noH, ligand_ag, idx_map)
    print(f"[info] Ligand negative ag_idx: {lig_neg_ag}   positive ag_idx: {lig_pos_ag}")

    n_mol = mol_noH.GetNumAtoms()
    occurrences: List[InteractionOccurrence] = []

    def _prot_atom_meta(prot_idx):
        pa = protein_ag.atoms[prot_idx]
        rn  = pa.resname.upper()
        rid = int(pa.resid)
        rc  = pa.segid.strip() or "?"
        return rn, rid, rc

    def _lig_mol_idx(ag_idx):
        mi = idx_map.get(int(ag_idx), 0)
        return min(mi, n_mol - 1)

    for ts in traj_slice:
        frame_idx = ts.frame
        prot_pos_coords = protein_ag.positions
        lig_pos_coords  = ligand_ag.positions
        seen_frame = set()

        # ── Protein NEG ↔ Ligand POS ─────────────────────────────────────────
        if prot_neg_idx and lig_pos_ag:
            pn_pos = prot_pos_coords[prot_neg_idx]    # (nPN, 3)
            lp_pos = lig_pos_coords[lig_pos_ag]        # (nLP, 3)
            d      = cdist(pn_pos, lp_pos)             # (nPN, nLP)
            hits   = np.argwhere(d <= cutoff)
            for pi_local, lj_local in hits:
                pi = prot_neg_idx[pi_local]
                lj = lig_pos_ag[lj_local]
                res_name, res_number, res_chain = _prot_atom_meta(pi)
                res_key = make_residue_key(res_name, res_number, res_chain)
                mi = _lig_mol_idx(lj)
                pct_key = f"{res_key}|Ionic|atom:{mi}"
                if (frame_idx, pct_key) in seen_frame: continue
                seen_frame.add((frame_idx, pct_key))
                occurrences.append(InteractionOccurrence(
                    frame=int(frame_idx), residue_key=res_key,
                    residue_name=res_name, residue_number=res_number,
                    residue_chain=res_chain, group="Ionic",
                    prolif_name="Anionic", percentage_key=pct_key,
                    anchor=Anchor("atom", (mi,), None,
                                  coords_2d[mi], f"atom:{mi}"),
                    meta={}, water_residues=(),
                ))

        # ── Protein POS ↔ Ligand NEG ─────────────────────────────────────────
        if prot_pos_idx and lig_neg_ag:
            pp_pos = prot_pos_coords[prot_pos_idx]   # (nPP, 3)
            ln_pos = lig_pos_coords[lig_neg_ag]       # (nLN, 3)
            d      = cdist(pp_pos, ln_pos)            # (nPP, nLN)
            hits   = np.argwhere(d <= cutoff)
            for pi_local, lj_local in hits:
                pi = prot_pos_idx[pi_local]
                lj = lig_neg_ag[lj_local]
                res_name, res_number, res_chain = _prot_atom_meta(pi)
                res_key = make_residue_key(res_name, res_number, res_chain)
                mi = _lig_mol_idx(lj)
                pct_key = f"{res_key}|Ionic|atom:{mi}"
                if (frame_idx, pct_key) in seen_frame: continue
                seen_frame.add((frame_idx, pct_key))
                occurrences.append(InteractionOccurrence(
                    frame=int(frame_idx), residue_key=res_key,
                    residue_name=res_name, residue_number=res_number,
                    residue_chain=res_chain, group="Ionic",
                    prolif_name="Cationic", percentage_key=pct_key,
                    anchor=Anchor("atom", (mi,), None,
                                  coords_2d[mi], f"atom:{mi}"),
                    meta={}, water_residues=(),
                ))

    print(f"[info] Ionic custom occurrences: {len(occurrences)}")
    return occurrences


# ── 3. Metal coordination ─────────────────────────────────────────────────────

def detect_metal(universe, ligand_ag, metal_ag, traj_slice,
                  coords_2d, mol_noH, idx_map,
                  cutoff: float = 2.8) -> List[InteractionOccurrence]:
    """
    Detect metal coordination: metal atom within `cutoff` Å of ligand N/O/S atom.
    Default cutoff 2.8 Å covers ZN-N(2.0-2.2), ZN-O(2.0-2.3), ZN-S(2.3-2.5).
    """
    if metal_ag is None or metal_ag.n_atoms == 0:
        return []
    print(f"[info] Running custom Metal detector (cutoff={cutoff} Å) …")

    # Ligand coordination atoms (N, O, S)
    lig_coord_idx = [i for i, a in enumerate(ligand_ag.atoms)
                     if _atom_element(a) in COORD_EL]
    if not lig_coord_idx:
        print("[warning] No ligand N/O/S atoms found; Metal detection skipped")
        return []

    lig_coord_arr = np.array(lig_coord_idx, dtype=int)
    n_mol = mol_noH.GetNumAtoms()
    occurrences: List[InteractionOccurrence] = []

    for ts in traj_slice:
        frame_idx = ts.frame
        metal_pos = metal_ag.positions                   # (nMetal, 3)
        lig_pos   = ligand_ag.positions[lig_coord_arr]   # (nCoord, 3)

        d = cdist(metal_pos, lig_pos)   # (nMetal, nCoord)
        hits = np.argwhere(d <= cutoff)
        seen_frame = set()

        for mi_local, lj_local in hits:
            ma = metal_ag.atoms[int(mi_local)]
            la_ag_idx = int(lig_coord_arr[lj_local])

            res_name   = ma.resname.upper()
            res_number = int(ma.resid)
            res_chain  = ma.segid.strip() or "?"
            res_key    = make_residue_key(res_name, res_number, res_chain)

            lig_mol_idx = idx_map.get(la_ag_idx, 0)
            lig_mol_idx = min(lig_mol_idx, n_mol - 1)

            pct_key = f"{res_key}|Metal|atom:{lig_mol_idx}"
            if (frame_idx, pct_key) in seen_frame:
                continue
            seen_frame.add((frame_idx, pct_key))

            occurrences.append(InteractionOccurrence(
                frame=int(frame_idx), residue_key=res_key,
                residue_name=res_name, residue_number=res_number,
                residue_chain=res_chain, group="Metal",
                prolif_name="MetalCoordination", percentage_key=pct_key,
                anchor=Anchor("atom", (lig_mol_idx,), None,
                              coords_2d[lig_mol_idx], f"atom:{lig_mol_idx}"),
                meta={}, water_residues=(),
            ))

    print(f"[info] Metal custom occurrences: {len(occurrences)}")
    return occurrences


# =============================================================================
# Aggregate occurrences
# =============================================================================

def aggregate_occurrences(occurrences: List[InteractionOccurrence],
                           n_frames: int,
                           min_percent: float,
                           per_group_min: Optional[Dict[str, float]] = None
                           ) -> List[AggregatedInteraction]:
    groups: Dict[str, List[InteractionOccurrence]] = defaultdict(list)
    for occ in occurrences:
        groups[occ.percentage_key].append(occ)

    results = []
    for pct_key, occs in groups.items():
        fc  = len({o.frame for o in occs})
        pct = 100.0 * fc / max(n_frames, 1)
        first = occs[0]
        # Use group-specific threshold if provided, else global
        grp_min = min_percent
        if per_group_min:
            grp_min = per_group_min.get(first.group, min_percent)
        if pct < grp_min:
            continue
        wo    = int(first.meta.get("order", 0)) if first.group == "WaterBridge" else 0
        results.append(AggregatedInteraction(
            residue_key=first.residue_key, residue_name=first.residue_name,
            residue_number=first.residue_number, residue_chain=first.residue_chain,
            group=first.group,
            prolif_names=sorted({o.prolif_name for o in occs}),
            percentage=pct, frame_count=fc, anchor=first.anchor,
            water_order=wo, water_residues=first.water_residues,
            metadata_examples=[o.meta for o in occs[:3]],
        ))

    results.sort(key=lambda x: (-x.percentage, x.residue_number,
                                x.residue_name, x.group))
    return results


# =============================================================================
# Node layout (even angular distribution)
# =============================================================================

def _ccw_gap(a, b): return (b - a) % (2 * math.pi)

def _angle_diff_signed(a: float, b: float) -> float:
    """Signed angular difference (b - a) in (-π, π]."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def spread_angles_evenly(natural, min_gap_deg=28.0, max_deviation_deg=65.0):
    """
    Spread nodes with a minimum gap while guaranteeing no node moves more than
    max_deviation_deg away from its natural (anchor-based) angle.

    If spreading would violate the deviation limit, the node is held at its
    natural angle and only Cartesian fine-tuning prevents overlap.
    """
    if len(natural) <= 1:
        return dict(natural)
    n  = len(natural)
    mg = math.radians(min_gap_deg)
    md = math.radians(max_deviation_deg)

    sk  = sorted(natural, key=lambda k: natural[k] % (2 * math.pi))
    nat = [natural[k] % (2 * math.pi) for k in sk]

    # If not enough room, use equal spacing anchored to mean direction
    if n * mg >= 2 * math.pi:
        step  = 2 * math.pi / n
        start = math.atan2(sum(math.sin(a) for a in nat),
                           sum(math.cos(a) for a in nat))
        return {k: start + i * step for i, k in enumerate(sk)}

    angles = list(nat)

    for _ in range(8000):
        changed = False
        for i in range(n):
            j   = (i + 1) % n
            gap = _ccw_gap(angles[i], angles[j])
            if gap < mg:
                push = (mg - gap) * 0.15
                # Before pushing, check deviation limit
                new_i = (angles[i] - push) % (2 * math.pi)
                new_j = (angles[j] + push) % (2 * math.pi)
                # Only apply push if within max deviation
                if abs(_angle_diff_signed(nat[i], new_i)) <= md:
                    angles[i] = new_i
                else:
                    angles[i] = (nat[i] - md) % (2 * math.pi)  # clamp
                if abs(_angle_diff_signed(nat[j], new_j)) <= md:
                    angles[j] = new_j
                else:
                    angles[j] = (nat[j] + md) % (2 * math.pi)  # clamp
                changed = True

        # Gentle pull back toward natural angle
        for i in range(n):
            diff    = _angle_diff_signed(angles[i], nat[i])
            angles[i] = (angles[i] + diff * 0.015) % (2 * math.pi)

        if not changed:
            break

    # Final enforcement: if any node still deviates beyond limit, hard-reset
    for i in range(n):
        if abs(_angle_diff_signed(nat[i], angles[i])) > md:
            angles[i] = nat[i]

    return {k: angles[i] for i, k in enumerate(sk)}


def build_residue_nodes(interactions, ligand_center, bbox,
                        min_gap_deg=28.0) -> Dict[str, ResidueNode]:
    """
    Place residue nodes on an outer ellipse.
    Steps:
    1. Compute natural angle (weighted anchor direction) per residue.
    2. Spread angles to enforce min_gap_deg while clamping deviation ≤ 90°
       (node cannot flip to the opposite side of the ligand).
    3. Cartesian fine-tuning for any residual overlap.
    """
    nodes  = {}
    by_res = defaultdict(list)
    for it in interactions:
        by_res[it.residue_key].append(it)

    xmin, xmax, ymin, ymax = bbox
    rx = (xmax - xmin) / 2.0 + 6.5
    ry = (ymax - ymin) / 2.0 + 5.8

    # Step 1: natural angles
    natural: Dict[str, float] = {}
    for key, its in by_res.items():
        w = np.array([max(it.percentage, 1e-3) for it in its])
        a = np.array([angle_of(it.anchor.coord - ligand_center) for it in its])
        natural[key] = math.atan2(float(np.dot(np.sin(a), w)),
                                   float(np.dot(np.cos(a), w)))

    # Step 2: spread with max 90° deviation (prevents opposite-side placement)
    spread = spread_angles_evenly(natural, min_gap_deg=min_gap_deg,
                                  max_deviation_deg=90.0)

    # Step 3: create nodes at spread positions
    for key, its in by_res.items():
        first = its[0]
        theta = spread[key]
        cat   = residue_category(first.residue_name)
        r     = 0.55 if cat == "water" else 0.78
        pos   = ligand_center + np.array([rx * math.cos(theta),
                                           ry * math.sin(theta)])
        nodes[key] = ResidueNode(
            key=key, name=first.residue_name, number=first.residue_number,
            chain=first.residue_chain, category=cat, radius=r,
            angle=theta, x=float(pos[0]), y=float(pos[1]),
        )

    # Step 4: Cartesian fine-tuning with 90° clamp
    _fine_tune_anchor_locked(nodes, ligand_center, rx, ry,
                             iterations=600, max_dev_deg=90.0)
    return nodes


def _fine_tune_anchor_locked(nodes, center, rx, ry,
                              iterations=800, max_dev_deg=70.0):
    """
    Push overlapping nodes apart in Cartesian space.
    After each push, clamp each node back within max_dev_deg of its
    natural (anchor-based) angle so it cannot end up on the wrong side.
    """
    nlist   = list(nodes.values())
    max_dev = math.radians(max_dev_deg)
    if len(nlist) < 2:
        return

    for _ in range(iterations):
        # Pairwise repulsion
        for i in range(len(nlist)):
            for j in range(i + 1, len(nlist)):
                a, b  = nlist[i], nlist[j]
                d     = np.array([a.x - b.x, a.y - b.y], dtype=float)
                dist  = float(np.linalg.norm(d)) + 1e-9
                mind  = a.radius + b.radius + 0.45
                if dist < mind:
                    push = (mind - dist) * 0.05
                    d   /= dist
                    a.x += float(d[0] * push); a.y += float(d[1] * push)
                    b.x -= float(d[0] * push); b.y -= float(d[1] * push)

        # Gentle pull toward ellipse + clamp to natural angle sector
        for n in nlist:
            # Current angle from center
            cur_angle = angle_of(np.array([n.x - float(center[0]),
                                            n.y - float(center[1])]))
            # Check deviation from natural anchor angle
            dev = _angle_diff_signed(n.angle, cur_angle)  # n.angle = natural
            if abs(dev) > max_dev:
                # Hard clamp: project back onto allowed sector boundary
                clamped = (n.angle + math.copysign(max_dev, dev)) % (2 * math.pi)
                cur_dist = math.hypot(n.x - float(center[0]),
                                      n.y - float(center[1]))
                r_eff = max(cur_dist, (rx + ry) / 2)  # keep roughly same radius
                n.x = float(center[0]) + r_eff * math.cos(clamped)
                n.y = float(center[1]) + r_eff * math.sin(clamped)
            else:
                # Gentle pull toward ellipse anchor
                tx = float(center[0]) + rx * math.cos(n.angle)
                ty = float(center[1]) + ry * math.sin(n.angle)
                n.x = 0.97 * n.x + 0.03 * tx
                n.y = 0.97 * n.y + 0.03 * ty

def _fine_tune(nodes, center, rx, ry, iterations=300):
    nlist = list(nodes.values())
    if len(nlist) < 2: return
    for _ in range(iterations):
        for i in range(len(nlist)):
            for j in range(i+1, len(nlist)):
                a, b = nlist[i], nlist[j]
                d = np.array([a.x-b.x, a.y-b.y], dtype=float)
                dist = float(np.linalg.norm(d)) + 1e-9
                mind = a.radius + b.radius + 0.38
                if dist < mind:
                    push = (mind-dist)*0.04; d /= dist
                    a.x += float(d[0]*push); a.y += float(d[1]*push)
                    b.x -= float(d[0]*push); b.y -= float(d[1]*push)
        for n in nlist:
            n.x = 0.97*n.x + 0.03*(float(center[0])+rx*math.cos(n.angle))
            n.y = 0.97*n.y + 0.03*(float(center[1])+ry*math.sin(n.angle))


# =============================================================================
# Drawing
# =============================================================================

def _grad_circle(ax, x, y, r, c_out, c_in, edge="#aaaaaa", zorder=5):
    for i in range(22, 0, -1):
        t  = i/22
        co = np.array(matplotlib.colors.to_rgb(c_out))
        ci = np.array(matplotlib.colors.to_rgb(c_in))
        ax.add_patch(Circle((x,y), r*(0.15+0.85*t),
                            facecolor=ci*(1-t)+co*t, edgecolor="none", zorder=zorder))
    ax.add_patch(Circle((x,y), r, facecolor="none",
                        edgecolor=edge, linewidth=0.9, zorder=zorder+0.2))

def draw_residue_node(ax, node: ResidueNode):
    co, ci = RESIDUE_NODE_COLORS.get(node.category, RESIDUE_NODE_COLORS["other"])
    _grad_circle(ax, node.x, node.y, node.radius, co, ci, zorder=6)
    fs = 8 if node.radius >= 0.75 else 7
    ax.text(node.x, node.y,
            pretty_residue_label(node.name, node.number, node.chain),
            ha="center", va="center", fontsize=fs, family="DejaVu Sans",
            color="#111111", fontweight="bold", zorder=7, linespacing=1.3)

def draw_water_node(ax, x, y, r=0.44):
    co, ci = RESIDUE_NODE_COLORS["water"]
    _grad_circle(ax, x, y, r, co, ci, zorder=8)
    ax.text(x, y, "H\u2082O", ha="center", va="center", fontsize=8,
            family="DejaVu Sans", color="#333333", fontweight="bold", zorder=9)

def _bond_offset(p1, p2, amount):
    vec = p2-p1; norm = np.linalg.norm(vec)
    return np.array([-vec[1],vec[0]])/norm*amount if norm>1e-9 else np.zeros(2)

def draw_ligand(ax, mol: Chem.Mol, coords: np.ndarray):
    dm = Chem.Mol(mol)
    try: Chem.Kekulize(dm, clearAromaticFlags=False)
    except Exception: pass
    for bond in dm.GetBonds():
        a,b   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p1,p2 = coords[a], coords[b]
        bt    = bond.GetBondType()
        col, lw = "#1a1a1a", 2.4
        if bt == Chem.rdchem.BondType.DOUBLE:
            off = _bond_offset(p1,p2,0.12)
            ax.plot([p1[0]+off[0],p2[0]+off[0]],[p1[1]+off[1],p2[1]+off[1]],
                    color=col,lw=2.0,solid_capstyle="round",zorder=2)
            ax.plot([p1[0]-off[0],p2[0]-off[0]],[p1[1]-off[1],p2[1]-off[1]],
                    color=col,lw=2.0,solid_capstyle="round",zorder=2)
        elif bt == Chem.rdchem.BondType.TRIPLE:
            off = _bond_offset(p1,p2,0.15)
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=col,lw=1.6,
                    solid_capstyle="round",zorder=2)
            for s in [1,-1]:
                ax.plot([p1[0]+s*off[0],p2[0]+s*off[0]],
                        [p1[1]+s*off[1],p2[1]+s*off[1]],
                        color=col,lw=1.2,solid_capstyle="round",zorder=2)
        else:
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                    color=col,lw=lw,solid_capstyle="round",zorder=2)

    ATOM_COLORS = {
        "O":"#cc1111","N":"#1133dd","S":"#b8960a","P":"#b86000",
        "F":"#1a8c3a","Cl":"#148844","Br":"#6e2800","I":"#5a10e0",
    }
    for atom in dm.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C": continue
        p = coords[atom.GetIdx()]
        ax.text(p[0],p[1],sym,fontsize=15,family="DejaVu Sans",
                fontweight="bold",color=ATOM_COLORS.get(sym,"#333333"),
                ha="center",va="center",zorder=4)

def _pct_label(ax, text, p1, p2, color, off_scale=0.22, zo=20):
    mid = midpoint(p1,p2); vec = p2-p1; norm = np.linalg.norm(vec)
    off = np.array([-vec[1],vec[0]])/norm*off_scale if norm>1e-9 else np.zeros(2)
    ax.text(mid[0]+off[0],mid[1]+off[1],text,fontsize=8.5,color=color,
            ha="center",va="center",fontweight="bold",zorder=zo)

def draw_direct_interaction(ax, it: AggregatedInteraction, node: ResidueNode):
    st    = INTERACTION_STYLES[it.group]
    start = np.array(it.anchor.coord, dtype=float)
    end   = np.array([node.x, node.y], dtype=float)
    delta = end - start; dist = np.linalg.norm(delta)
    end2  = end - delta/dist*node.radius*0.95 if dist>1e-8 else end
    ax.plot([start[0],end2[0]],[start[1],end2[1]],
            color=st["color"],lw=st["linewidth"],linestyle=st["linestyle"],
            solid_capstyle="round",zorder=3,alpha=0.93)
    if st["ligand_dot"]:
        ax.add_patch(Circle((start[0],start[1]),0.08,
                            facecolor=st["color"],edgecolor="none",zorder=4))
    _pct_label(ax,f"{it.percentage:.0f}%",start,end2,st["label_color"])

def _build_water_node_positions(wb_interactions: List[AggregatedInteraction],
                                 ligand_center: np.ndarray,
                                 bbox,
                                 residue_nodes: Dict[str, ResidueNode],
                                 min_gap_deg: float = 32.0
                                 ) -> Dict[str, np.ndarray]:
    """
    Place H2O node at 55% of the way from the ligand anchor toward the
    residue node.  This guarantees the drawing order is always:
        ligand_anchor ──── [H2O] ──── residue_node
    with everything on the same straight line.

    If multiple water bridges share the same residue node, each gets a small
    perpendicular offset so they don't overlap.
    """
    if not wb_interactions:
        return {}

    def _wb_key(it: AggregatedInteraction) -> str:
        return f"{it.residue_key}|{it.anchor.key}"

    # Count how many WBs share each residue so we can offset them
    from collections import Counter
    residue_count: Dict[str, int] = Counter(it.residue_key for it in wb_interactions)
    residue_idx:   Dict[str, int] = defaultdict(int)

    positions: Dict[str, np.ndarray] = {}
    for it in wb_interactions:
        node = residue_nodes.get(it.residue_key)
        if node is None:
            continue

        start = np.array(it.anchor.coord, dtype=float)
        end   = np.array([node.x, node.y], dtype=float)
        delta = end - start
        dist  = np.linalg.norm(delta)

        # Base position: 55 % along anchor → residue
        base  = start + delta * 0.55

        # Perpendicular offset for multiple WBs to same residue
        idx   = residue_idx[it.residue_key]
        n_wb  = residue_count[it.residue_key]
        if n_wb > 1 and dist > 1e-8:
            perp   = np.array([-delta[1], delta[0]]) / dist
            spread = (idx - (n_wb - 1) / 2.0) * 0.35
            base   = base + perp * spread
        residue_idx[it.residue_key] += 1

        positions[_wb_key(it)] = base

    return positions


def draw_waterbridge(ax, it: AggregatedInteraction,
                     node: ResidueNode,
                     water_pos: np.ndarray):
    """
    ligand_anchor ──── [H2O] ──── residue_node
    H2O sits on inner ellipse, spread evenly around the ligand.
    """
    st    = INTERACTION_STYLES["WaterBridge"]
    start = np.array(it.anchor.coord, dtype=float)
    w     = water_pos.copy()
    end   = np.array([node.x, node.y], dtype=float)

    # Segment 1: ligand anchor → H2O
    ax.plot([start[0], w[0]], [start[1], w[1]],
            color=st["color"], lw=st["linewidth"],
            linestyle=st["linestyle"],
            solid_capstyle="round", zorder=3, alpha=0.93)
    _pct_label(ax, f"{it.percentage:.0f}%",
               start, w, st["label_color"], off_scale=0.20)

    # Segment 2: H2O → residue (stop at node boundary)
    d2    = end - w
    dist2 = np.linalg.norm(d2)
    end2  = end - d2 / dist2 * node.radius * 0.95 if dist2 > 1e-8 else end
    ax.plot([w[0], end2[0]], [w[1], end2[1]],
            color=st["color"], lw=st["linewidth"],
            linestyle=st["linestyle"],
            solid_capstyle="round", zorder=3, alpha=0.93)
    _pct_label(ax, f"{it.percentage:.0f}%",
               w, end2, st["label_color"], off_scale=0.20)

    # H2O node on top
    draw_water_node(ax, float(w[0]), float(w[1]))



def draw_title(ax, title, bbox, nodes):
    if not title: return
    xmin,xmax,_,_ = bbox
    top = max(n.y+n.radius for n in nodes.values()) + 1.4
    ax.text((xmin+xmax)/2.0, top, title, ha="center", va="bottom",
            fontsize=18, color="#1a1a2e", family="DejaVu Sans",
            fontweight="bold", zorder=50)

def draw_legend(ax, groups_present, xmin, ymin):
    x0, y0, dy = xmin+0.15, ymin+0.15, 0.68
    seen_labels = set()
    row = 0
    for grp in groups_present:
        st    = INTERACTION_STYLES.get(grp, {})
        label = st.get("legend", grp)
        if label in seen_labels:
            continue          # skip duplicate legend entries (e.g. PiStacking = Hydrophobic)
        seen_labels.add(label)
        ax.plot([x0, x0+0.85], [y0+row*dy]*2,
                color=st.get("color","#888"), lw=st.get("linewidth",1.5),
                linestyle=st.get("linestyle","solid"),
                solid_capstyle="round", zorder=50)
        if st.get("ligand_dot"):
            ax.add_patch(Circle((x0, y0+row*dy), 0.07,
                                facecolor=st["color"], edgecolor="none", zorder=51))
        ax.text(x0+1.05, y0+row*dy, label,
                fontsize=9, va="center", color="#1a1a1a",
                fontweight="bold", zorder=50)
        row += 1

def final_limits(coords, nodes, extra=2.8):
    xs = list(coords[:,0]); ys = list(coords[:,1])
    for n in nodes.values():
        xs += [n.x-n.radius, n.x+n.radius]
        ys += [n.y-n.radius, n.y+n.radius]
    return min(xs)-extra, max(xs)+extra, min(ys)-extra, max(ys)+extra


# =============================================================================
# Export
# =============================================================================

def export_csv(path, interactions):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["residue_chain","residue_name","residue_number",
                    "interaction_group","prolif_types","percentage",
                    "frame_count","anchor_kind","anchor_key","water_order"])
        for it in interactions:
            w.writerow([it.residue_chain,it.residue_name,it.residue_number,
                        it.group,";".join(it.prolif_names),
                        f"{it.percentage:.3f}",it.frame_count,
                        it.anchor.kind,it.anchor.key,it.water_order])

def export_json(path, interactions):
    data = [{
        "residue_key":it.residue_key,"residue_name":it.residue_name,
        "residue_number":it.residue_number,"residue_chain":it.residue_chain,
        "group":it.group,"prolif_names":it.prolif_names,
        "percentage":it.percentage,"frame_count":it.frame_count,
        "water_order":it.water_order,"water_residues":list(it.water_residues),
        "anchor":{"kind":it.anchor.kind,"atom_indices":list(it.anchor.atom_indices),
                  "ring_index":it.anchor.ring_index,
                  "coord":[float(it.anchor.coord[0]),float(it.anchor.coord[1])],
                  "key":it.anchor.key},
    } for it in interactions]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_interactive_html(path: Path,
                             mol: Chem.Mol,
                             coords: np.ndarray,
                             interactions: List[AggregatedInteraction],
                             nodes: Dict[str, ResidueNode],
                             water_positions: Dict[str, np.ndarray],
                             title: str = ""):
    """
    Fully interactive HTML:
    - Pan/zoom the whole diagram (scroll + drag background)
    - Drag any residue node, water node, OR the ligand individually
    - Lines update live when anything moves
    - ↺ Layout resets all positions
    Architecture: CSS transform on SVG for pan/zoom.
    All element positions stored in base-screen-pixels (no double scaling).
    """
    ATOM_COLORS_JS = {
        "O": "#cc1111", "N": "#1133dd", "S": "#b8960a",
        "P": "#b86000", "F": "#1a8c3a", "Cl": "#148844",
        "Br": "#6e2800", "I": "#5a10e0",
    }

    draw_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(draw_mol, clearAromaticFlags=False)
    except Exception:
        pass

    bonds_data = []
    for bond in draw_mol.GetBonds():
        a, b  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt    = bond.GetBondType()
        btype = ("double" if bt == Chem.rdchem.BondType.DOUBLE
                 else "triple" if bt == Chem.rdchem.BondType.TRIPLE
                 else "single")
        bonds_data.append({
            "a": a, "b": b, "type": btype,
            "x1": float(coords[a][0]), "y1": float(coords[a][1]),
            "x2": float(coords[b][0]), "y2": float(coords[b][1]),
        })

    atoms_data = []
    for atom in draw_mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C": continue
        i = atom.GetIdx()
        atoms_data.append({"sym": sym, "x": float(coords[i][0]),
                            "y": float(coords[i][1]),
                            "color": ATOM_COLORS_JS.get(sym, "#333333")})

    nodes_data = []
    for key, n in nodes.items():
        co, ci = RESIDUE_NODE_COLORS.get(n.category, RESIDUE_NODE_COLORS["other"])
        nodes_data.append({
            "id": key, "label": f"{n.name}\n{n.number}",
            "x": n.x, "y": n.y, "r": n.radius,
            "colorOuter": co, "colorInner": ci,
        })

    water_data = []
    for wkey, wpos in water_positions.items():
        water_data.append({
            "id": f"water_{wkey}",
            "x": float(wpos[0]), "y": float(wpos[1]),
        })

    lines_data = []
    for it in interactions:
        st   = INTERACTION_STYLES[it.group]
        wkey = f"{it.residue_key}|{it.anchor.key}"
        is_wb = it.group == "WaterBridge" and wkey in water_positions
        lines_data.append({
            "group":       it.group,
            "color":       st["color"],
            "lw":          st["linewidth"],
            "dash":        _linestyle_to_dasharray(st["linestyle"]),
            "dot":         st["ligand_dot"],
            "anchorX":     float(it.anchor.coord[0]),
            "anchorY":     float(it.anchor.coord[1]),
            "nodeId":      it.residue_key,
            "pct":         f"{it.percentage:.0f}%",
            "isWB":        is_wb,
            "waterNodeId": f"water_{wkey}" if is_wb else None,
        })

    legend_data = []
    seen_lbl: set = set()
    for it in interactions:
        st  = INTERACTION_STYLES[it.group]
        lbl = st.get("legend", it.group)
        if lbl not in seen_lbl:
            seen_lbl.add(lbl)
            legend_data.append({
                "label": lbl, "color": st["color"],
                "lw": st["linewidth"],
                "dash": _linestyle_to_dasharray(st["linestyle"]),
                "dot": st["ligand_dot"],
            })

    diagram_data = {
        "title":  title or "Ligand Interaction Map",
        "bonds":  bonds_data, "atoms":  atoms_data,
        "nodes":  nodes_data, "waters": water_data,
        "lines":  lines_data, "legend": legend_data,
    }
    js_data = json.dumps(diagram_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title or 'Ligand Interaction Map'} — Interactive</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#e8e8e8;font-family:Arial,sans-serif;overflow:hidden}}
#toolbar{{position:fixed;top:8px;left:50%;transform:translateX(-50%);
  background:rgba(255,255,255,0.97);border-radius:8px;
  box-shadow:0 2px 14px rgba(0,0,0,0.18);padding:5px 14px;
  display:flex;align-items:center;gap:8px;z-index:300;user-select:none}}
#toolbar button{{background:#3498db;color:#fff;border:none;border-radius:5px;
  padding:4px 12px;cursor:pointer;font-size:13px;font-weight:bold;
  transition:background 0.15s}}
#toolbar button:hover{{background:#1a6fa8}}
#toolbar button.warn{{background:#e67e22}}
#toolbar button.warn:hover{{background:#b8610a}}
#lbl{{font-size:12px;color:#555;min-width:52px;text-align:center}}
#hint{{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,0.5);color:#fff;padding:4px 16px;
  border-radius:20px;font-size:11px;pointer-events:none;z-index:200}}
#canvas{{width:100vw;height:100vh;background:white;overflow:hidden;
  position:relative;cursor:grab}}
#canvas.panning{{cursor:grabbing}}
svg{{position:absolute;left:0;top:0;width:100%;height:100%;
  transform-origin:0 0;}}
.node-group{{cursor:grab}}
.node-group:hover > circle:first-child{{stroke-width:2px;stroke:#444}}
.node-group.dragging{{cursor:grabbing}}
#lig-group{{cursor:grab}}
#lig-group.dragging{{cursor:grabbing}}
</style>
</head>
<body>
<div id="toolbar">
  <button onclick="zoomBtn(1.25)">＋ Zoom</button>
  <button onclick="zoomBtn(0.8)">－ Zoom</button>
  <button onclick="resetView()">⌂ View</button>
  <button class="warn" onclick="resetLayout()">↺ Reset</button>
  <span id="lbl">100%</span>
  <span style="color:#888;font-size:11px">Drag: ligand · residues · H₂O</span>
</div>
<div id="hint">Scroll = zoom · Background drag = pan · Node/Ligand drag = reposition · ↺ = reset layout</div>
<div id="canvas">
  <svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg>
</div>

<script>
const DATA = {js_data};

// ─── Coordinate system ────────────────────────────────────────────────────
// All diagram coords (from Python) → screen base-pixels via sx/sy
// Pan+zoom applied as CSS transform on SVG (no double-transform)

let allX=[], allY=[];
DATA.bonds.forEach(b=>{{ allX.push(b.x1,b.x2); allY.push(b.y1,b.y2); }});
DATA.nodes.forEach(n=>{{ allX.push(n.x-n.r,n.x+n.r); allY.push(n.y-n.r,n.y+n.r); }});
DATA.waters.forEach(w=>{{ allX.push(w.x,w.x); allY.push(w.y,w.y); }});
DATA.lines.forEach(l=>{{ allX.push(l.anchorX); allY.push(l.anchorY); }});

const DX=Math.min(...allX)-3, DY=Math.min(...allY)-3;
const DW=Math.max(...allX)-DX+6, DH=Math.max(...allY)-DY+6;
const W=window.innerWidth, H=window.innerHeight;
const SCALE = Math.min(W/DW, H/DH) * 0.90;
const OX = (W - DW*SCALE)/2;
const OY = (H - DH*SCALE)/2;

// diagram → base-screen-pixel
function bx(x){{ return (x-DX)*SCALE + OX; }}
function by(y){{ return (y-DY)*SCALE + OY; }}
function br(r){{ return r*SCALE; }}

// ─── Pan / zoom state (CSS transform on SVG) ─────────────────────────────
let sc=1, ptx=0, pty=0;

function svgTransform(){{
  svg.style.transform = `translate(${{ptx}}px,${{pty}}px) scale(${{sc}})`;
  document.getElementById('lbl').textContent = Math.round(sc*100)+'%';
}}

function zoomAt(mx,my,f){{
  // mx,my in screen pixels; zoom around that point
  ptx = mx - (mx-ptx)*f;
  pty = my - (my-pty)*f;
  sc  = Math.min(Math.max(sc*f, 0.05), 30);
  svgTransform();
  updateLines();   // refresh label sizes
}}
function zoomBtn(f){{ zoomAt(W/2, H/2, f); }}
function resetView(){{ sc=1; ptx=0; pty=0; svgTransform(); updateLines(); }}

// ─── Ligand offset (base-screen-pixels) ──────────────────────────────────
let ligDx=0, ligDy=0;

// ─── Live node positions (base-screen-pixels) ────────────────────────────
const nodePos={{}};
DATA.nodes.forEach(n=>{{ nodePos[n.id]={{x:bx(n.x), y:by(n.y)}}; }});
const waterPos={{}};
DATA.waters.forEach(w=>{{ waterPos[w.id]={{x:bx(w.x), y:by(w.y)}}; }});

function resetLayout(){{
  ligDx=0; ligDy=0;
  gLig.setAttribute('transform','translate(0,0)');
  DATA.nodes.forEach(n=>{{ nodePos[n.id]={{x:bx(n.x),y:by(n.y)}}; }});
  DATA.waters.forEach(w=>{{ waterPos[w.id]={{x:bx(w.x),y:by(w.y)}}; }});
  applyNodePositions();
  updateLines();
}}

// ─── SVG helpers ─────────────────────────────────────────────────────────
const svg  = document.getElementById('svg');
const canvas = document.getElementById('canvas');

function el(tag,cls){{
  const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
  if(cls) e.setAttribute('class',cls);
  return e;
}}
function sa(e,attrs){{
  for(const[k,v] of Object.entries(attrs)) e.setAttribute(k,String(v));
  return e;
}}

// ─── Layers ───────────────────────────────────────────────────────────────
const gLines  = el('g','lines-layer');   svg.appendChild(gLines);
const gLig    = el('g','lig-layer');     svg.appendChild(gLig);
const gWater  = el('g','water-layer');   svg.appendChild(gWater);
const gNodes  = el('g','nodes-layer');   svg.appendChild(gNodes);
const gLegend = el('g','legend-layer'); svg.appendChild(gLegend);

// ─── Radial gradient defs (match SVG/PNG look exactly) ───────────────────
const defs = el('defs'); svg.insertBefore(defs, svg.firstChild);

function makeRadialGrad(id, colorOuter, colorInner){{
  const g = el('radialGradient');
  g.setAttribute('id', id);
  g.setAttribute('cx','35%'); g.setAttribute('cy','35%');
  g.setAttribute('r','65%');
  g.setAttribute('fx','35%'); g.setAttribute('fy','35%');
  const s1=el('stop'); s1.setAttribute('offset','0%');
  s1.setAttribute('stop-color', colorInner);
  const s2=el('stop'); s2.setAttribute('offset','100%');
  s2.setAttribute('stop-color', colorOuter);
  g.appendChild(s1); g.appendChild(s2);
  defs.appendChild(g);
}}

// One gradient per unique colorOuter+colorInner pair
const gradMap={{}};
function gradId(co,ci){{
  const key=co+ci;
  if(!gradMap[key]){{
    const id='grad_'+Object.keys(gradMap).length;
    gradMap[key]=id;
    makeRadialGrad(id,co,ci);
  }}
  return gradMap[key];
}}

// Pre-create water gradient
makeRadialGrad('grad_water','#bdc3c7','#f8f9fa');

// Pre-create all node gradients
DATA.nodes.forEach(n=>{{ gradId(n.colorOuter, n.colorInner); }});

gLig.setAttribute('id','lig-group');

// ─── Draw ligand ──────────────────────────────────────────────────────────
DATA.bonds.forEach(b=>{{
  const lw=2.4*SCALE/60;
  const x1=bx(b.x1),y1=by(b.y1),x2=bx(b.x2),y2=by(b.y2);
  if(b.type==='double'){{
    const off=0.12*SCALE, dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1;
    const px=-dy/len*off, py=dx/len*off;
    [[px,py],[-px,-py]].forEach(([ox,oy])=>{{
      gLig.appendChild(sa(el('line'),{{x1:x1+ox,y1:y1+oy,x2:x2+ox,y2:y2+oy,
        stroke:'#1a1a1a','stroke-width':lw*0.85,'stroke-linecap':'round'}}));
    }});
  }} else if(b.type==='triple'){{
    const off=0.15*SCALE, dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1;
    const px=-dy/len*off, py=dx/len*off;
    [[0,0],[px,py],[-px,-py]].forEach(([ox,oy],i)=>{{
      gLig.appendChild(sa(el('line'),{{x1:x1+ox,y1:y1+oy,x2:x2+ox,y2:y2+oy,
        stroke:'#1a1a1a','stroke-width':lw*(i?0.7:0.85),'stroke-linecap':'round'}}));
    }});
  }} else {{
    gLig.appendChild(sa(el('line'),{{x1,y1,x2,y2,
      stroke:'#1a1a1a','stroke-width':lw,'stroke-linecap':'round'}}));
  }}
}});

DATA.atoms.forEach(a=>{{
  const t=sa(el('text'),{{x:bx(a.x),y:by(a.y),'text-anchor':'middle',
    'dominant-baseline':'central','font-size':15*SCALE/60,
    'font-weight':'bold','font-family':'Arial',fill:a.color}});
  t.textContent=a.sym;
  gLig.appendChild(t);
}});

// ─── Interaction lines (dynamic) ─────────────────────────────────────────
const lineEls=[];
DATA.lines.forEach(ld=>{{
  const g=el('g'); gLines.appendChild(g);
  const mkLine=()=>sa(el('line'),{{stroke:ld.color,
    'stroke-width':ld.lw*SCALE/60,'stroke-dasharray':ld.dash,
    'stroke-linecap':'round',opacity:0.93}});
  const l1=mkLine(); g.appendChild(l1);
  const l2=ld.isWB ? (g.appendChild(mkLine()),g.lastChild) : null;
  const dot=ld.dot ? sa(el('circle'),{{r:0.08*SCALE,fill:ld.color}}) : null;
  if(dot) g.appendChild(dot);
  const mkT=()=>{{
    const t=sa(el('text'),{{'text-anchor':'middle','dominant-baseline':'central',
      'font-size':8.5*SCALE/60,'font-weight':'bold',fill:ld.color}});
    g.appendChild(t); return t;
  }};
  const t1=mkT(), t2=ld.isWB?mkT():null;
  lineEls.push({{l1,l2,dot,t1,t2,ld}});
}});

// ─── Water nodes ──────────────────────────────────────────────────────────
const waterEls={{}};
DATA.waters.forEach(w=>{{
  const g=sa(el('g'),{{'class':'node-group'}}); gWater.appendChild(g);
  const r=br(0.44);
  const c1=sa(el('circle'),{{r,
    fill:'url(#grad_water)',
    stroke:'#aaaaaa','stroke-width':0.8}});
  const t=sa(el('text'),{{'text-anchor':'middle','dominant-baseline':'central',
    'font-size':8*SCALE/60,'font-weight':'bold','font-family':'Arial',
    fill:'#333333','pointer-events':'none'}});
  t.textContent='H₂O';
  g.append(c1,t);
  waterEls[w.id]={{g}};
}});

// ─── Residue nodes ────────────────────────────────────────────────────────
const nodeEls={{}};
DATA.nodes.forEach(n=>{{
  const g=sa(el('g'),{{'class':'node-group'}}); gNodes.appendChild(g);
  const r=br(n.r);
  const gid='url(#'+gradId(n.colorOuter,n.colorInner)+')';
  const bg=sa(el('circle'),{{r, fill:gid,
    stroke:'#aaaaaa','stroke-width':0.9}});
  const t=sa(el('text'),{{'text-anchor':'middle','dominant-baseline':'central',
    'font-size':8*SCALE/60,'font-weight':'bold','font-family':'Arial',
    fill:'#111111','pointer-events':'none'}});
  const lns=n.label.split('\\n'), dy=8.5*SCALE/60;
  const y0=-(lns.length-1)*dy/2;
  lns.forEach((ln,i)=>{{
    const ts=el('tspan');
    ts.setAttribute('x','0');
    ts.setAttribute('dy', i===0 ? y0+'px' : dy+'px');
    ts.textContent=ln;
    t.appendChild(ts);
  }});
  g.append(bg,t);
  nodeEls[n.id]={{g}};
}});

// ─── Legend ───────────────────────────────────────────────────────────────
const legX=12, legY0=H-14-(DATA.legend.length*20);
DATA.legend.forEach((lg,i)=>{{
  const y=legY0+i*20;
  gLegend.appendChild(sa(el('line'),{{x1:legX,y1:y,x2:legX+28,y2:y,
    stroke:lg.color,'stroke-width':lg.lw*SCALE/60,
    'stroke-dasharray':lg.dash,'stroke-linecap':'round'}}));
  if(lg.dot) gLegend.appendChild(sa(el('circle'),
    {{cx:legX,cy:y,r:4,fill:lg.color}}));
  const lt=sa(el('text'),{{x:legX+34,y,
    'dominant-baseline':'central','font-size':11,'font-weight':'bold',
    'font-family':'Arial',fill:'#222'}});
  lt.textContent=lg.label;
  gLegend.appendChild(lt);
}});

if(DATA.title){{
  const tt=sa(el('text'),{{x:W/2,y:22,'text-anchor':'middle',
    'font-size':17,'font-weight':'bold','font-family':'Arial',fill:'#1a1a2e'}});
  tt.textContent=DATA.title;
  gLegend.appendChild(tt);
}}

// ─── Position helpers ─────────────────────────────────────────────────────
function applyNodePositions(){{
  DATA.nodes.forEach(n=>{{
    const p=nodePos[n.id];
    nodeEls[n.id].g.setAttribute('transform',`translate(${{p.x}},${{p.y}})`);
  }});
  DATA.waters.forEach(w=>{{
    const p=waterPos[w.id];
    waterEls[w.id].g.setAttribute('transform',`translate(${{p.x}},${{p.y}})`);
  }});
}}

function updateLines(){{
  lineEls.forEach(({{l1,l2,dot,t1,t2,ld}})=>{{
    // Anchor in base pixels, shifted by ligand offset
    const ax=bx(ld.anchorX)+ligDx, ay=by(ld.anchorY)+ligDy;
    const np=nodePos[ld.nodeId];
    const nx=np.x, ny=np.y;

    // Shorten to node boundary
    const ddx=nx-ax, ddy=ny-ay, dd=Math.hypot(ddx,ddy)||1;
    const rn=br(DATA.nodes.find(n=>n.id===ld.nodeId)?.r||0.78);
    const ex=nx-ddx/dd*rn*0.95, ey=ny-ddy/dd*rn*0.95;

    if(ld.isWB && ld.waterNodeId && waterPos[ld.waterNodeId]){{
      const wp=waterPos[ld.waterNodeId];
      sa(l1,{{x1:ax,y1:ay,x2:wp.x,y2:wp.y}});
      sa(l2,{{x1:wp.x,y1:wp.y,x2:ex,y2:ey}});
      pct(t1,ax,ay,wp.x,wp.y,ld.pct);
      pct(t2,wp.x,wp.y,ex,ey,ld.pct);
    }} else {{
      sa(l1,{{x1:ax,y1:ay,x2:ex,y2:ey}});
      pct(t1,ax,ay,ex,ey,ld.pct);
    }}
    if(dot) sa(dot,{{cx:ax,cy:ay}});
  }});
}}

function pct(te,x1,y1,x2,y2,txt){{
  const mx=(x1+x2)/2, my=(y1+y2)/2;
  const dx=x2-x1, dy=y2-y1, len=Math.hypot(dx,dy)||1;
  const off=0.22*SCALE;
  te.setAttribute('x', mx-dy/len*off);
  te.setAttribute('y', my+dx/len*off);
  te.textContent=txt;
  te.setAttribute('font-size', 8.5*SCALE/60);
}}

// ─── Drag state ───────────────────────────────────────────────────────────
// type: 'pan' | 'node' | 'water' | 'ligand'
let drag=null;

function screenToBase(sx,sy){{
  // Convert screen pixels to base-screen-pixels (undo CSS transform)
  return {{ x:(sx-ptx)/sc, y:(sy-pty)/sc }};
}}

// Ligand drag
gLig.addEventListener('mousedown',e=>{{
  e.stopPropagation();
  const b=screenToBase(e.clientX,e.clientY);
  drag={{type:'ligand', sx:b.x, sy:b.y, ox:ligDx, oy:ligDy}};
  gLig.classList.add('dragging');
}});

// Node drag
Object.entries(nodeEls).forEach(([id,{{g}}])=>{{
  g.addEventListener('mousedown',e=>{{
    e.stopPropagation();
    const b=screenToBase(e.clientX,e.clientY);
    const p=nodePos[id];
    drag={{type:'node', id, sx:b.x, sy:b.y, ox:p.x, oy:p.y}};
    g.classList.add('dragging');
  }});
}});

// Water drag
Object.entries(waterEls).forEach(([id,{{g}}])=>{{
  g.addEventListener('mousedown',e=>{{
    e.stopPropagation();
    const b=screenToBase(e.clientX,e.clientY);
    const p=waterPos[id];
    drag={{type:'water', id, sx:b.x, sy:b.y, ox:p.x, oy:p.y}};
    g.classList.add('dragging');
  }});
}});

// Background drag = pan
canvas.addEventListener('mousedown',e=>{{
  if(!drag){{
    drag={{type:'pan', sx:e.clientX, sy:e.clientY, opx:ptx, opy:pty}};
    canvas.classList.add('panning');
  }}
}});

window.addEventListener('mousemove',e=>{{
  if(!drag) return;
  if(drag.type==='pan'){{
    ptx=drag.opx+(e.clientX-drag.sx);
    pty=drag.opy+(e.clientY-drag.sy);
    svgTransform();
    updateLines();
  }} else {{
    const b=screenToBase(e.clientX,e.clientY);
    const ddx=b.x-drag.sx, ddy=b.y-drag.sy;
    if(drag.type==='ligand'){{
      ligDx=drag.ox+ddx; ligDy=drag.oy+ddy;
      gLig.setAttribute('transform',`translate(${{ligDx}},${{ligDy}})`);
      updateLines();
    }} else if(drag.type==='node'){{
      nodePos[drag.id]={{x:drag.ox+ddx, y:drag.oy+ddy}};
      nodeEls[drag.id].g.setAttribute('transform',
        `translate(${{nodePos[drag.id].x}},${{nodePos[drag.id].y}})`);
      updateLines();
    }} else if(drag.type==='water'){{
      waterPos[drag.id]={{x:drag.ox+ddx, y:drag.oy+ddy}};
      waterEls[drag.id].g.setAttribute('transform',
        `translate(${{waterPos[drag.id].x}},${{waterPos[drag.id].y}})`);
      updateLines();
    }}
  }}
}});

window.addEventListener('mouseup',()=>{{
  if(drag){{
    if(drag.type==='ligand') gLig.classList.remove('dragging');
    else if(drag.type==='node') nodeEls[drag.id]?.g.classList.remove('dragging');
    else if(drag.type==='water') waterEls[drag.id]?.g.classList.remove('dragging');
    else canvas.classList.remove('panning');
    drag=null;
  }}
}});

// Scroll = zoom
canvas.addEventListener('wheel',e=>{{
  e.preventDefault();
  zoomAt(e.clientX,e.clientY, e.deltaY<0?1.12:0.88);
}},{{passive:false}});

canvas.addEventListener('dblclick',resetView);

// ─── Init ─────────────────────────────────────────────────────────────────
applyNodePositions();
updateLines();
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def _linestyle_to_dasharray(ls) -> str:
    """Convert matplotlib linestyle to SVG stroke-dasharray string."""
    if ls == "solid" or ls is None:
        return "none"
    if isinstance(ls, tuple) and len(ls) == 2:
        offset, pattern = ls
        if isinstance(pattern, (list, tuple)):
            return " ".join(str(round(v * 4, 1)) for v in pattern)
    return "none"


# =============================================================================
# Protein-Ligand Contacts Histogram (Schrödinger/Desmond-style stacked bars)
# =============================================================================

# Stacking order, bottom → top (matches reference layout convention)
HISTOGRAM_GROUP_ORDER = [
    "Hydrophobic", "HydrogenBond", "PiCation",
    "Ionic", "Metal", "Halogen", "WaterBridge",
]


def compute_residue_group_fractions(
        occurrences: List[InteractionOccurrence],
        n_frames: int
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Tuple[str, int, str]]]:
    """
    For the histogram, merge across ALL anchor atoms per residue.
    fraction[(residue_key, group)] = (number of distinct frames in which
    AT LEAST ONE occurrence of that residue+group exists) / n_frames.

    A residue can have multiple interaction types active in the same frame,
    so the stacked sum per residue may exceed 1.0 — this matches the
    standard "Interactions Fraction" convention.
    """
    frames_by_rg: Dict[Tuple[str, str], set] = defaultdict(set)
    res_info: Dict[str, Tuple[str, int, str]] = {}
    for occ in occurrences:
        frames_by_rg[(occ.residue_key, occ.group)].add(occ.frame)
        res_info[occ.residue_key] = (occ.residue_chain, occ.residue_number,
                                      occ.residue_name)
    fractions = {k: len(v) / max(n_frames, 1) for k, v in frames_by_rg.items()}
    return fractions, res_info


def draw_contacts_histogram(svg_path: Path, png_path: Path,
                             fractions: Dict[Tuple[str, str], float],
                             res_info: Dict[str, Tuple[str, int, str]],
                             min_fraction: float = 0.01,
                             title: str = "Protein\u2013Ligand Contacts",
                             width: float = 14.0, height: float = 5.0,
                             dpi: int = 600) -> bool:
    """
    Stacked bar chart: one bar per residue, stacked by interaction-type
    fraction.  Bottom\u2192top order = HISTOGRAM_GROUP_ORDER.
    X-axis: residue number only (rotated \u221245\u00b0).  Y-axis: Interactions Fraction.
    Closed black border box, white background \u2014 matches the classic
    Schr\u00f6dinger/Desmond simulation-interaction-diagram histogram layout.
    """
    # Residues that pass the threshold in ANY group
    residues = sorted(
        {rk for (rk, g), f in fractions.items() if f >= min_fraction},
        key=lambda rk: (res_info[rk][0], res_info[rk][1])
    )
    if not residues:
        print("[warning] No residues meet histogram threshold "
              f"(>={min_fraction*100:.1f}%); skipping contacts histogram.")
        return False

    n = len(residues)
    fig_w = max(width, n * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, height), dpi=dpi)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    x = np.arange(n)
    bottom = np.zeros(n)
    bar_width = 0.80
    legend_handles = []

    for group in HISTOGRAM_GROUP_ORDER:
        vals = np.array([fractions.get((rk, group), 0.0) for rk in residues])
        if vals.sum() <= 0:
            continue
        style = INTERACTION_STYLES[group]
        bars = ax.bar(x, vals, bottom=bottom, width=bar_width,
                      color=style["color"], edgecolor="none",
                      label=style.get("legend", group), zorder=3)
        legend_handles.append(bars)
        bottom += vals

    # X-axis: residue NAME + NUMBER (e.g. "HIS129"), rotated to avoid overlap
    ax.set_xticks(x)
    labels = [f"{res_info[rk][2].upper()}{res_info[rk][1]}" for rk in residues]
    ax.set_xticklabels(labels, rotation=-60, ha="right",
                       fontsize=7.5, family="DejaVu Sans")
    ax.set_xlim(-0.6, n - 0.4)

    # Y-axis
    ymax = max(float(bottom.max()) * 1.15, 0.1)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Interactions Fraction", fontsize=11,
                  fontweight="bold", family="DejaVu Sans")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                    family="DejaVu Sans", pad=10)

    # Closed border box (matches reference: all 4 spines visible)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)

    if legend_handles:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92,
                  ncol=1, edgecolor="#888888")

    fig.tight_layout()
    fig.savefig(svg_path, facecolor="white")
    fig.savefig(png_path, facecolor="white", dpi=dpi)
    plt.close(fig)
    return True



# =============================================================================
# FULL MD REPORT — GROMACS automation + Schrodinger-style RMSD/RMSF plots
# =============================================================================

SID_COLOR_PROTEIN = "#4a738c"   # steel blue  - protein / Cα RMSD line
SID_COLOR_LIGAND  = "#e60000"   # red - ligand RMSD line
SID_COLOR_SS_BAND = "#cfe8f3"   # pale blue   - secondary structure shading
SID_COLOR_CONTACT = "#6fbf73"   # green       - ligand-contact residue shading
SID_LINE_GRAY     = "#5a6b73"   # neutral dark gray-blue for RMSF curve


# =============================================================================
# Generic .xvg reader (handles GROMACS comment/legend lines)
# =============================================================================

def read_xvg(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 2-column (or more) GROMACS .xvg file -> (x, y) arrays.
    If more than 2 data columns exist, only the first y column is returned."""
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(xs), np.array(ys)


def read_xvg_multi(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Read .xvg with multiple y columns -> (x, [y1, y2, ...])."""
    xs: List[float] = []
    ys_cols: List[List[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            xs.append(vals[0])
            if not ys_cols:
                ys_cols = [[] for _ in vals[1:]]
            for i, v in enumerate(vals[1:]):
                ys_cols[i].append(v)
    return np.array(xs), [np.array(c) for c in ys_cols]


# =============================================================================
# GROMACS command runner
# =============================================================================

class GmxError(RuntimeError):
    pass


def run_gmx(gmx_bin: str, args: List[str], stdin_text: str,
           cwd: Optional[Path] = None, label: str = "") -> str:
    """
    Run a `gmx <tool>` command, sending stdin_text (group selections,
    newline separated) to stdin.  Raises GmxError with full stderr on failure.
    """
    cmd = [gmx_bin] + args
    print(f"[gmx] {label or args[0]}: {' '.join(cmd)}  (stdin={stdin_text!r})")
    try:
        result = subprocess.run(
            cmd, input=stdin_text, capture_output=True, text=True,
            cwd=str(cwd) if cwd else None, timeout=3600,
        )
    except FileNotFoundError:
        raise GmxError(f"GROMACS binary '{gmx_bin}' not found in PATH. "
                       f"Activate your GROMACS environment first.")
    except subprocess.TimeoutExpired:
        raise GmxError(f"gmx {args[0]} timed out after 1 hour.")

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-25:])
        raise GmxError(f"gmx {args[0]} failed (exit {result.returncode}).\n"
                       f"--- stderr tail ---\n{tail}")
    return result.stdout


def gmx_rms(gmx_bin: str, topol: Path, traj: Path, out: Path,
           group_fit: int, group_rmsd: int, cwd: Path,
           extra_args: Optional[List[str]] = None) -> bool:
    args = ["rms", "-s", str(topol), "-f", str(traj), "-o", str(out), "-tu", "ns"]
    if extra_args:
        args += extra_args
    try:
        run_gmx(gmx_bin, args, f"{group_fit}\n{group_rmsd}\n", cwd=cwd,
               label=f"RMSD (fit={group_fit}, calc={group_rmsd})")
        return True
    except GmxError as e:
        print(f"[error] {e}")
        return False


def gmx_rmsf(gmx_bin: str, topol: Path, traj: Path, out: Path,
            group: int, cwd: Path, per_residue: bool = False) -> bool:
    args = ["rmsf", "-s", str(topol), "-f", str(traj), "-o", str(out)]
    if per_residue:
        args.append("-res")
    try:
        run_gmx(gmx_bin, args, f"{group}\n", cwd=cwd,
               label=f"RMSF (group={group}, res={per_residue})")
        return True
    except GmxError as e:
        print(f"[error] {e}")
        return False


def gmx_hbond(gmx_bin: str, topol: Path, traj: Path, out_num: Path,
             group1: int, group2: int, cwd: Path) -> bool:
    # GROMACS >= 2023 renamed `gmx hbond` to `gmx hbond` (kept) — try both
    # historical names defensively.
    for tool in ["hbond", "h-bond", "hbond-legacy"]:
        args = [tool, "-s", str(topol), "-f", str(traj), "-num", str(out_num)]
        try:
            run_gmx(gmx_bin, args, f"{group1}\n{group2}\n", cwd=cwd,
                   label=f"H-bond count ({tool})")
            return True
        except GmxError as e:
            last_err = e
            continue
    print(f"[error] All hbond tool name variants failed. Last error: {last_err}")
    return False


def gmx_gyrate(gmx_bin: str, topol: Path, traj: Path, out: Path,
               group: int, cwd: Path) -> bool:
    args = ["gyrate", "-s", str(topol), "-f", str(traj), "-o", str(out)]
    try:
        run_gmx(gmx_bin, args, f"{group}\n", cwd=cwd, label="Radius of gyration")
        return True
    except GmxError as e:
        print(f"[error] {e}")
        return False


def gmx_make_combined_index(gmx_bin: str, topol: Path, out_ndx: Path,
                            cwd: Path,
                            group_a: int = 1, group_b: int = 13) -> bool:
    """
    Run `gmx make_ndx` to create a merged 'Protein_LIG' style group via
    `<a> | <b>`, needed by `gmx sasa -surface Protein_LIG -output LIG`.
    """
    args = ["make_ndx", "-f", str(topol), "-o", str(out_ndx)]
    stdin_text = f"{group_a} | {group_b}\nq\n"
    try:
        run_gmx(gmx_bin, args, stdin_text, cwd=cwd,
               label="make_ndx (merge Protein|LIG)")
        return True
    except GmxError as e:
        print(f"[error] {e}")
        return False


def gmx_sasa(gmx_bin: str, topol: Path, traj: Path, out: Path,
            index_file: Path, cwd: Path,
            surface_group: str = "Protein_LIG",
            output_group: str = "LIG") -> bool:
    args = ["sasa", "-f", str(traj), "-s", str(topol), "-n", str(index_file),
            "-o", str(out), "-surface", surface_group, "-output", output_group]
    try:
        run_gmx(gmx_bin, args, "", cwd=cwd, label="SASA")
        return True
    except GmxError as e:
        print(f"[error] {e}")
        return False


def try_gmx_dssp_secondary_structure(
        gmx_bin: str, topol: Path, traj: Path, cwd: Path
) -> Optional[Dict[int, str]]:
    """
    Best-effort secondary-structure assignment per residue using `gmx dssp`.
    Returns {residue_number: dominant_ss_code} or None if unavailable.
    Never raises — any failure just disables the SS shading gracefully.
    """
    out_dat = cwd / "_ss_temp.dat"
    args = ["dssp", "-s", str(topol), "-f", str(traj), "-o", str(out_dat)]
    try:
        run_gmx(gmx_bin, args, "", cwd=cwd, label="Secondary structure (dssp)")
    except GmxError as e:
        print(f"[info] Secondary structure unavailable ({e}); "
              f"continuing without SS shading.")
        return None

    if not out_dat.exists():
        return None

    # `gmx dssp` output: one line per frame, one character per residue
    # (H,G,I = helix; E,B = sheet; else coil/turn/loop)
    try:
        lines = [l.strip() for l in out_dat.read_text().splitlines()
                if l.strip() and not l.startswith(("#", "@"))]
        if not lines:
            return None
        n_res = len(lines[0])
        from collections import Counter
        counts = [Counter() for _ in range(n_res)]
        for line in lines:
            for i, ch in enumerate(line[:n_res]):
                counts[i][ch] += 1
        dominant = {}
        for i, c in enumerate(counts):
            if c:
                dominant[i + 1] = c.most_common(1)[0][0]
        return dominant
    except Exception as e:
        print(f"[info] Could not parse dssp output ({e}); skipping SS shading.")
        return None
    finally:
        out_dat.unlink(missing_ok=True)


# =============================================================================
# PLOT 1 — Protein-Ligand RMSD (dual-axis, Schrödinger style)
# =============================================================================

def draw_rmsd_dual_axis(svg_path: Path, png_path: Path,
                        time_prot: np.ndarray, rmsd_prot: np.ndarray,
                        time_lig: np.ndarray, rmsd_lig: np.ndarray,
                        dpi: int = 300) -> None:
    """
    Recreates the classic Schrödinger/Desmond "Protein-Ligand RMSD" panel:
    - Left axis (black):  Protein Cα RMSD, steel-blue line
    - Right axis (red):   Ligand RMSD fit-on-protein, red line + red label
    - Legend ("Cα" / "(Lig) fit on Prot") centered just above the plot
    - X-axis: Time (nsec)
    """
    fig, ax1 = plt.subplots(figsize=(10.88, 6.87), dpi=dpi)
    fig.subplots_adjust(left=0.115, right=0.88, bottom=0.110, top=0.880)

    l1, = ax1.plot(time_prot, rmsd_prot, color=SID_COLOR_PROTEIN,
                   linewidth=1.5, solid_capstyle="projecting", alpha=0.9)
    ax1.set_xlabel("Time (nsec)", fontsize=13, family="DejaVu Sans")
    ax1.set_ylabel("Protein RMSD (\u00c5)", fontsize=13, family="DejaVu Sans",
                   color="black")
    ax1.tick_params(axis="both", labelsize=11)
    ax1.set_xlim(float(time_prot.min()) if len(time_prot) else 0,
                float(time_prot.max()) if len(time_prot) else 1)
    y1max = max(float(rmsd_prot.max()) * 1.15, 1.0) if len(rmsd_prot) else 1.0
    ax1.set_ylim(0, y1max)

    ax2 = ax1.twinx()
    l2, = ax2.plot(time_lig, rmsd_lig, color=SID_COLOR_LIGAND,
                   linewidth=1.5, solid_capstyle="projecting", alpha=0.9)
    ax2.tick_params(axis="y", labelsize=11, colors=SID_COLOR_LIGAND)
    ax2.set_ylabel("Ligand RMSD (\u00c5)", fontsize=13, family="DejaVu Sans",
                   color=SID_COLOR_LIGAND, labelpad=8)
    y2max = max(float(rmsd_lig.max()) * 1.15, 1.0) if len(rmsd_lig) else 1.0
    ax2.set_ylim(0, y2max)
    for spine in ("right",):
        ax2.spines[spine].set_color(SID_COLOR_LIGAND)

    # Legend centered above the plot (matches reference position)
    leg = fig.legend(
        [l1, l2], ["C\u03b1", "(Lig) fit on Prot"],
        loc="upper center", bbox_to_anchor=(0.5125, 0.935),
        ncol=2, fontsize=11, frameon=True, framealpha=0.8,
        edgecolor="#cccccc", handlelength=1.6, columnspacing=1.4,
    )
    leg.get_frame().set_linewidth(0.8)

    ax1.spines["top"].set_visible(True)
    fig.savefig(svg_path, facecolor="white")
    fig.savefig(png_path, facecolor="white", dpi=dpi)
    plt.close(fig)


# =============================================================================
# PLOT 2 — Protein RMSF with secondary-structure + ligand-contact shading
# =============================================================================

def draw_protein_rmsf(svg_path: Path, png_path: Path,
                      residue_nums: np.ndarray, rmsf: np.ndarray,
                      contact_residues: Optional[set] = None,
                      ss_assignment: Optional[Dict[int, str]] = None,
                      dpi: int = 300, width: float = 14.0,
                      height: float = 4.5) -> None:
    """
    Schrödinger-style "Protein RMSF" (matches reference image exactly):
    - Title: "Protein RMSF"
    - Blue Cα RMSF line, legend entry "Cα"
    - X-axis: Residue Index, starts from 0 (or actual first residue)
    - Y-axis: RMSF (Å)
    - Pale-blue vertical bands at helices/sheets (from gmx dssp if available)
    - GREEN VERTICAL STEMS from y=0 up to the RMSF curve value at each
      ligand-contact residue (Schrödinger style: individual vlines, NOT fill)
    - No top/right spines
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.88)

    xmin_data = float(residue_nums.min())
    xmax_data = float(residue_nums.max())
    # X-axis: always starts at 0 (or the first actual residue if it is < 0)
    xmin_plot = 0.0
    ymax = max(float(rmsf.max()) * 1.18, 0.5)

    # ── 1. Secondary-structure shading (pale blue bands) ────────────────────
    if ss_assignment:
        in_band = False
        band_start = None
        for rn in range(int(xmin_data), int(xmax_data) + 2):
            is_ss = ss_assignment.get(rn, "-") in ("H", "G", "I", "E", "B")
            if is_ss and not in_band:
                in_band = True
                band_start = rn - 0.5
            elif not is_ss and in_band:
                in_band = False
                ax.axvspan(band_start, rn - 0.5, color=SID_COLOR_SS_BAND,
                           zorder=0, linewidth=0)
        if in_band:
            ax.axvspan(band_start, xmax_data + 0.5, color=SID_COLOR_SS_BAND,
                       zorder=0, linewidth=0)

    # ── 2. Green vertical stems at ligand-contact residues (Schrödinger style)
    #       Each stem is a vertical line from y=0 to the RMSF value at that
    #       residue — drawn BEFORE the main curve so the blue line sits on top.
    if contact_residues:
        # Build a lookup: residue_number -> rmsf_value
        rmsf_lookup = {int(rn): float(rv)
                       for rn, rv in zip(residue_nums, rmsf)}
        for rn in sorted(contact_residues):
            if rn in rmsf_lookup:
                rv = rmsf_lookup[rn]
                ax.vlines(rn, 0, rv,
                          colors=SID_COLOR_CONTACT,
                          linewidth=2.5, zorder=2, alpha=0.95)

    # ── 3. Main RMSF curve (blue, Cα) ───────────────────────────────────────
    line_ca, = ax.plot(residue_nums, rmsf,
                       color=SID_COLOR_PROTEIN,   # steel blue
                       linewidth=1.4, zorder=3, label="C\u03b1")

    # ── 4. Axes limits & labels ──────────────────────────────────────────────
    ax.set_xlim(xmin_plot, xmax_data + 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Residue Index", fontsize=12, family="DejaVu Sans")
    ax.set_ylabel("RMSF (\u00c5)", fontsize=12, family="DejaVu Sans")
    ax.set_title("Protein RMSF", fontsize=14, fontweight="bold",
                 family="DejaVu Sans", pad=8)

    # ── 5. X-ticks: round numbers, always include 0 and the last residue ────
    n_res = int(xmax_data - xmin_plot) + 1
    step  = max(1, round(n_res / 25))
    # Round step to nearest 10/50/100 for clean labels
    if step >= 100:
        step = round(step / 100) * 100
    elif step >= 50:
        step = round(step / 50) * 50
    elif step >= 10:
        step = round(step / 10) * 10

    tick_vals = list(range(0, int(xmax_data) + 1, step))
    if tick_vals[-1] != int(xmax_data):
        tick_vals.append(int(xmax_data))
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(v) for v in tick_vals], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    # ── 6. Legend & spines ───────────────────────────────────────────────────
    ax.legend(handles=[line_ca], loc="upper right",
              fontsize=10, frameon=True, framealpha=0.85,
              edgecolor="#cccccc", handlelength=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(png_path, facecolor="white", dpi=dpi)
    plt.close(fig)


# =============================================================================
# PLOT 4 — Ligand RMSF with 2D structure on top (fixes x-tick crowding)
# =============================================================================

LIG_ELEMENT_COLORS = {
    "N": "#0000FF", "O": "#FF0000", "S": "#B8860B", "F": "#00BB00",
    "Cl": "#00BB00", "Br": "#8B2500", "I": "#660099", "P": "#FF8000",
    "C": "#000000",
}


def _shorten(xi, yi, xj, yj, s):
    dx, dy = xj - xi, yj - yi
    L = math.hypot(dx, dy)
    if L == 0:
        return xi, yi, xj, yj
    ux, uy = dx / L, dy / L
    return xi + ux * s, yi + uy * s, xj - ux * s, yj - uy * s


def _perp_toward_center(x1, y1, x2, y2, cx, cy, d):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dcx, dcy = cx - mx, cy - my
    L = math.hypot(dcx, dcy)
    return (0, 0) if L == 0 else ((dcx / L) * d, (dcy / L) * d)


def _perp_offset(x1, y1, x2, y2, d):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    return (0, 0) if L == 0 else ((-dy / L) * d, (dx / L) * d)


def draw_ligand_mol_on_ax(ax, mol, coords, font_scale: float = 1.0):
    """Draw 2D ligand structure with atom-number labels (heavy atoms only)."""
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    bond_rings = ring_info.BondRings()

    ring_centers = []
    for ring in atom_rings:
        ring_centers.append((
            np.mean([coords[i][0] for i in ring]),
            np.mean([coords[i][1] for i in ring])
        ))
    bond_to_ring = {}
    for ri, rbonds in enumerate(bond_rings):
        for b in rbonds:
            bond_to_ring.setdefault(b, ri)

    C, LW, S, IS, D = "#222222", 2.0, 0.30, 0.22, 0.16

    for bond in mol.GetBonds():
        bidx = bond.GetIdx()
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        xi, yi = coords[i]; xj, yj = coords[j]
        btype = bond.GetBondTypeAsDouble()
        xs, ys, xe, ye = _shorten(xi, yi, xj, yj, S)
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        ux, uy = (dx / L, dy / L) if L > 0 else (0, 0)
        in_ring = bidx in bond_to_ring

        if btype == 1.0:
            ax.plot([xs, xe], [ys, ye], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
        elif btype == 2.0 and in_ring:
            ri = bond_to_ring[bidx]
            cx, cy = ring_centers[ri]
            ax.plot([xs, xe], [ys, ye], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
            ox, oy = _perp_toward_center(xs, ys, xe, ye, cx, cy, D)
            ax.plot([xs+ox+ux*IS, xe+ox-ux*IS], [ys+oy+uy*IS, ye+oy-uy*IS],
                    color=C, lw=LW, solid_capstyle="round", zorder=1)
        elif btype == 2.0:
            ox, oy = _perp_offset(xs, ys, xe, ye, D/2)
            ax.plot([xs+ox, xe+ox], [ys+oy, ye+oy], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
            ax.plot([xs-ox, xe-ox], [ys-oy, ye-oy], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
        elif btype == 3.0:
            ox, oy = _perp_offset(xs, ys, xe, ye, D*0.6)
            ax.plot([xs, xe], [ys, ye], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
            ax.plot([xs+ox, xe+ox], [ys+oy, ye+oy], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
            ax.plot([xs-ox, xe-ox], [ys-oy, ye-oy], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)
        else:
            ax.plot([xs, xe], [ys, ye], color=C, lw=LW,
                    solid_capstyle="round", zorder=1)

    all_x = [v[0] for v in coords.values()]
    all_y = [v[1] for v in coords.values()]
    scale = max(max(all_x)-min(all_x), max(all_y)-min(all_y), 1e-6)
    fs = max(8.0, min(14.0, 110/scale)) * font_scale

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        color = LIG_ELEMENT_COLORS.get(atom.GetSymbol(), "#000000")
        x, y = coords[idx]
        ax.text(x, y, str(idx+1), color=color, fontsize=fs, fontweight="bold",
                ha="center", va="center", zorder=3)

    margin = 0.6
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_y)-margin, max(all_y)+margin)
    ax.set_aspect("equal")
    ax.axis("off")
    return (max(all_x)-min(all_x)+2*margin), (max(all_y)-min(all_y)+2*margin)


def draw_ligand_rmsf(svg_path: Path, png_path: Path,
                     mol, coords: Dict[int, Tuple[float, float]],
                     rmsf_values: np.ndarray,
                     dpi: int = 300, fig_width: float = 14.0) -> None:
    """
    Ligand RMSF with 2D structure panel on top.
    - Molecule panel height is capped to avoid bloating for small ligands.
    - RMSF values are converted nm → Å if needed.
    - X-axis labels use adaptive font + rotation to avoid overlap.
    """
    # nm → Å conversion (gmx rmsf outputs nm)
    rmsf_values = np.array(rmsf_values, dtype=float)
    if rmsf_values.max() < 5.0:
        rmsf_values = rmsf_values * 10.0

    n_atoms = len(rmsf_values)
    atom_nums = np.arange(1, n_atoms + 1)

    all_x = [v[0] for v in coords.values()]
    all_y = [v[1] for v in coords.values()]
    bbox_w = (max(all_x) - min(all_x)) + 1.2
    bbox_h = (max(all_y) - min(all_y)) + 1.2

    left, right = 0.06, 0.985
    axis_width_in = fig_width * (right - left)

    # ── Molecule panel height: scale with aspect ratio but cap tightly ──────
    # For small ligands (< 25 atoms) cap at 5 inches — prevents huge empty panels.
    # For larger ligands allow up to 10 inches.
    raw_mol_h = axis_width_in * (bbox_h / bbox_w)
    if n_atoms <= 25:
        mol_h = float(np.clip(raw_mol_h, 3.0, 5.5))
    elif n_atoms <= 45:
        mol_h = float(np.clip(raw_mol_h, 4.0, 8.0))
    else:
        mol_h = float(np.clip(raw_mol_h, 4.5, 10.0))

    title_pad = 0.55
    rmsf_h = 4.4

    top, bottom = 0.96, 0.13
    fig_h = (mol_h + title_pad + rmsf_h) / (top - bottom)

    fig = plt.figure(figsize=(fig_width, fig_h), facecolor="white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[mol_h+title_pad, rmsf_h],
                           hspace=0.015, top=top, bottom=bottom,
                           left=left, right=right)

    ax_mol = fig.add_subplot(gs[0])
    ax_mol.set_title("Ligand RMSF", fontsize=18, fontweight="bold", pad=8)
    draw_ligand_mol_on_ax(ax_mol, mol, coords)

    ax = fig.add_subplot(gs[1])
    ax.plot(atom_nums, rmsf_values, color="#8B1A4A", linewidth=2.2,
           solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(0.5, n_atoms + 0.5)
    y_max = float(np.ceil(rmsf_values.max())) + 0.5
    ax.set_ylim(0, y_max)

    # ── Crowding fix: adaptive font size + rotation + tick thinning ──────────
    if n_atoms <= 20:
        fs, rot, step = 9.0, 0, 1
    elif n_atoms <= 35:
        fs, rot, step = 7.5, 90, 1
    elif n_atoms <= 60:
        fs, rot, step = 6.5, 90, 1
    else:
        fs, rot, step = 6.0, 90, 2

    shown_ticks = atom_nums[::step]
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels([str(i) for i in shown_ticks],
                       fontsize=fs, rotation=rot,
                       ha="center" if rot == 0 else "center",
                       va="top" if rot == 0 else "center_baseline")
    ax.tick_params(axis="x", pad=6 if rot else 2)

    ax.set_yticks(np.arange(0, int(y_max)+1, max(1, int(y_max)//6) or 1))
    ax.tick_params(axis="both", labelsize=10.5, length=4)
    ax.set_ylabel("RMSF (\u00c5)", fontsize=13, labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[Patch(facecolor="#8B1A4A", edgecolor="#8B1A4A",
                             label="Fit Ligand on Protein")],
             loc="upper right", fontsize=10.5, frameon=True,
             framealpha=0.9, edgecolor="#aaaaaa",
             handlelength=1.6, handleheight=1.0)

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(png_path, facecolor="white", dpi=dpi)
    plt.close(fig)


def draw_simple_timeseries(svg_path: Path, png_path: Path,
                           x: np.ndarray, y: np.ndarray,
                           xlabel: str, ylabel: str,
                           color: str = SID_COLOR_PROTEIN,
                           dpi: int = 300, width: float = 10.0,
                           height: float = 4.0) -> None:
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.93)
    ax.plot(x, y, color=color, linewidth=1.4)
    ax.set_xlabel(xlabel, fontsize=12, family="DejaVu Sans")
    ax.set_ylabel(ylabel, fontsize=12, family="DejaVu Sans")
    ax.tick_params(axis="both", labelsize=10)
    if len(x):
        ax.set_xlim(float(x.min()), float(x.max()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(svg_path, facecolor="white")
    fig.savefig(png_path, facecolor="white", dpi=dpi)
    plt.close(fig)


# =============================================================================
# ORCHESTRATION — run all GROMACS commands + generate all SID-style plots
# =============================================================================

def run_full_md_report(args, contact_residues: Optional[set] = None) -> Dict[str, str]:
    """
    Fully automated MD analysis report, using ONLY topol/traj/ligand-ref that
    were already supplied for the interaction diagram.  Runs:

      gmx rms     (protein Cα RMSD)       groups: fit / fit
      gmx rms     (ligand RMSD on prot)   groups: fit / calc
      gmx rmsf    (protein, per-residue)  group:  rmsf-protein
      gmx rmsf    (ligand)                group:  rmsf-ligand
      gmx hbond   (protein-ligand count)  groups: hbond1 / hbond2
      gmx gyrate  (protein Rg)            group:  gyrate
      gmx make_ndx + gmx sasa             (Protein_LIG / LIG)

    Then draws Schrodinger/Desmond-style figures (SVG + PNG) for each.
    contact_residues: residue numbers already identified as ligand-contacting
    by the main interaction-diagram analysis — reused to shade the Protein
    RMSF plot, so the two analyses are visually and scientifically linked.

    Any individual GROMACS step that fails prints a warning and is skipped;
    the rest of the report still completes.
    """
    topol = Path(args.topol).resolve()
    traj  = Path(args.traj).resolve()
    report_dir = Path(args.report_dir) if args.report_dir else Path(args.output_prefix).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_dir = report_dir.resolve()
    gmx_bin = args.gmx_bin

    print("\n" + "=" * 70)
    print("[report] Starting full automated MD report (GROMACS + SID plots)")
    print(f"[report] Output directory: {report_dir}")
    print("=" * 70)

    produced: Dict[str, str] = {}

    # ── 1. Protein RMSD (Cα) ────────────────────────────────────────────────
    rmsd_ca_xvg = report_dir / "rmsd_ca.xvg"
    ok_rmsd_prot = gmx_rms(gmx_bin, topol, traj, rmsd_ca_xvg,
                          args.rmsd_protein_g1, args.rmsd_protein_g2,
                          cwd=report_dir)

    # ── 2. Ligand RMSD (fit on protein) ─────────────────────────────────────
    rmsd_lig_xvg = report_dir / "rmsd_ligand.xvg"
    ok_rmsd_lig = gmx_rms(gmx_bin, topol, traj, rmsd_lig_xvg,
                         args.rmsd_ligand_g1, args.rmsd_ligand_g2,
                         cwd=report_dir)

    if ok_rmsd_prot and ok_rmsd_lig:
        try:
            t_p, y_p = read_xvg(rmsd_ca_xvg)
            t_l, y_l = read_xvg(rmsd_lig_xvg)
            # GROMACS rms outputs RMSD in nm regardless of -tu flag.
            # Convert nm → Å (multiply by 10).
            if y_p.max() < 5.0:   # safety check: already in Å if > 5
                y_p = y_p * 10.0
            if y_l.max() < 5.0:
                y_l = y_l * 10.0
            svg_o = report_dir / "PL-RMSD.svg"
            png_o = report_dir / "PL-RMSD.png"
            draw_rmsd_dual_axis(svg_o, png_o, t_p, y_p, t_l, y_l, dpi=args.dpi)
            produced["rmsd_svg"] = str(svg_o)
            produced["rmsd_png"] = str(png_o)
            print(f"[report] \u2713 Protein-Ligand RMSD plot saved: {png_o}")
        except Exception as e:
            print(f"[warning] Could not draw RMSD plot: {e}")
    else:
        print("[warning] Skipping RMSD plot (one or both gmx rms runs failed).")

    # ── 3. Protein RMSF (per-residue) ───────────────────────────────────────
    rmsf_prot_xvg = report_dir / "rmsf_protein.xvg"
    ok_rmsf_prot = gmx_rmsf(gmx_bin, topol, traj, rmsf_prot_xvg,
                           args.rmsf_protein_group, cwd=report_dir,
                           per_residue=True)
    if ok_rmsf_prot:
        try:
            res_idx_raw, rmsf_vals = read_xvg(rmsf_prot_xvg)
            res_idx_raw = res_idx_raw.astype(int)
            # gmx rmsf outputs in nm — convert to Å
            if rmsf_vals.max() < 5.0:
                rmsf_vals = rmsf_vals * 10.0
            n_res = len(res_idx_raw)

            # ── Re-index residues to sequential 1..N ────────────────────────
            # GROMACS may output residues numbered 800..1168 (from PDB
            # numbering). We renumber to 1..N so the X-axis always starts at 1.
            # We also build a mapping orig_num → sequential so contact_residues
            # (which carry ProLIF/MDAnalysis original numbers) can be translated.
            seq_res_idx = np.arange(1, n_res + 1)          # 1, 2, 3 … N
            orig_to_seq = {int(orig): int(seq)
                           for orig, seq in zip(res_idx_raw, seq_res_idx)}

            # Map contact residues from original PDB numbers → sequential idx
            contact_seq: Optional[set] = None
            if contact_residues:
                contact_seq = {orig_to_seq[r]
                               for r in contact_residues
                               if r in orig_to_seq}
                if not contact_seq:
                    # Fallback: if none mapped, maybe ProLIF already gave
                    # sequential numbers — keep them if they fall in range
                    contact_seq_fallback = {r for r in contact_residues
                                            if 1 <= r <= n_res}
                    contact_seq = contact_seq_fallback or None
                print(f"[info] Protein RMSF: {n_res} residues, "
                      f"original range {res_idx_raw[0]}–{res_idx_raw[-1]} "
                      f"→ renumbered 1–{n_res}. "
                      f"Contact stems: {len(contact_seq) if contact_seq else 0}")

            # Remap ss_assignment keys if needed
            ss_assignment = None
            if args.secondary_structure:
                ss_raw = try_gmx_dssp_secondary_structure(
                    gmx_bin, topol, traj, cwd=report_dir)
                if ss_raw:
                    # If ss keys are in original numbering, remap them too
                    if ss_raw and min(ss_raw.keys()) > n_res:
                        ss_assignment = {orig_to_seq[k]: v
                                         for k, v in ss_raw.items()
                                         if k in orig_to_seq}
                    else:
                        ss_assignment = ss_raw

            svg_o = report_dir / "Protein_RMSF.svg"
            png_o = report_dir / "Protein_RMSF.png"
            draw_protein_rmsf(svg_o, png_o, seq_res_idx, rmsf_vals,
                             contact_residues=contact_seq,
                             ss_assignment=ss_assignment, dpi=args.dpi)
            produced["protein_rmsf_svg"] = str(svg_o)
            produced["protein_rmsf_png"] = str(png_o)
            print(f"[report] \u2713 Protein RMSF plot saved: {png_o}")
        except Exception as e:
            print(f"[warning] Could not draw protein RMSF plot: {e}")
    else:
        print("[warning] Skipping protein RMSF plot (gmx rmsf failed).")

    # ── 4. Ligand RMSF (with 2D structure panel) ────────────────────────────
    rmsf_lig_xvg = report_dir / "rmsf_ligand.xvg"
    ok_rmsf_lig = gmx_rmsf(gmx_bin, topol, traj, rmsf_lig_xvg,
                          args.rmsf_ligand_group, cwd=report_dir,
                          per_residue=False)
    if ok_rmsf_lig:
        try:
            _, rmsf_lig_vals = read_xvg(rmsf_lig_xvg)
            if rmsf_lig_vals.max() < 5.0:          # nm -> Å conversion
                rmsf_lig_vals = rmsf_lig_vals * 10.0

            # ── Filter to heavy atoms ONLY ───────────────────────────────────
            # gmx rmsf for group "LIG" includes ALL atoms (heavy + H).
            # A 45-heavy-atom ligand gives 111 values if it has 66 hydrogens.
            # We detect heavy atoms from the topology and keep only those.
            try:
                _u_lig = mda.Universe(str(topol))
                _lig_ag = _u_lig.select_atoms(args.ligand_selection)
                _heavy_mask = np.array([a.mass > 1.5 for a in _lig_ag.atoms])
                n_heavy = int(_heavy_mask.sum())
                n_xvg   = len(rmsf_lig_vals)
                if n_xvg > n_heavy and n_heavy > 0:
                    # Build index array of heavy atom positions among all atoms
                    heavy_idx_arr = np.where(_heavy_mask)[0]
                    # Only apply if sizes are consistent
                    if len(heavy_idx_arr) <= n_xvg:
                        rmsf_lig_vals = rmsf_lig_vals[heavy_idx_arr]
                        print(f"[info] Ligand RMSF: filtered "
                              f"{n_xvg} atoms → {len(rmsf_lig_vals)} "
                              f"heavy atoms (removed H)")
                    else:
                        # Fallback: truncate to heavy count if indices mismatch
                        rmsf_lig_vals = rmsf_lig_vals[:n_heavy]
                        print(f"[info] Ligand RMSF: truncated to "
                              f"{n_heavy} heavy atoms")
                else:
                    print(f"[info] Ligand RMSF: {n_xvg} values, "
                          f"{n_heavy} heavy atoms — no filtering needed")
            except Exception as _e:
                print(f"[info] Ligand RMSF heavy-atom filter skipped: {_e}")

            if args.ligand_ref and Path(args.ligand_ref).exists():
                lig_mol = load_ligand_mol2_for_rmsf(args.ligand_ref)
                if lig_mol is not None:
                    lig_mol2, lig_coords = lig_mol
                    n_mol2  = lig_mol2.GetNumAtoms()
                    n_rmsf  = len(rmsf_lig_vals)
                    if n_mol2 != n_rmsf:
                        print(f"[info] Ligand RMSF: mol2 has {n_mol2} heavy "
                              f"atoms, rmsf has {n_rmsf} values — "
                              f"using min({n_mol2},{n_rmsf}) for curve")
                        # Use shorter of the two for the RMSF curve
                        rmsf_lig_vals = rmsf_lig_vals[:min(n_mol2, n_rmsf)]
                    svg_o = report_dir / "Ligand_RMSF.svg"
                    png_o = report_dir / "Ligand_RMSF.png"
                    draw_ligand_rmsf(svg_o, png_o, lig_mol2, lig_coords,
                                    rmsf_lig_vals, dpi=args.dpi)
                    produced["ligand_rmsf_svg"] = str(svg_o)
                    produced["ligand_rmsf_png"] = str(png_o)
                    print(f"[report] \u2713 Ligand RMSF plot saved: {png_o}")
                else:
                    print("[warning] Could not load --ligand-ref for RMSF "
                         "structure panel; skipping ligand RMSF plot.")
            else:
                print("[warning] No --ligand-ref (.mol2) provided; "
                     "skipping ligand RMSF structure panel.")
        except Exception as e:
            print(f"[warning] Could not draw ligand RMSF plot: {e}")
    else:
        print("[warning] Skipping ligand RMSF plot (gmx rmsf failed).")

    # ── 5. Radius of gyration ────────────────────────────────────────────────
    gyrate_xvg = report_dir / "gyrate.xvg"
    if gmx_gyrate(gmx_bin, topol, traj, gyrate_xvg,
                 args.gyrate_group, cwd=report_dir):
        try:
            t, rg = read_xvg(gyrate_xvg)
            if rg.max() < 20.0:   # nm → Å
                rg = rg * 10.0
            svg_o = report_dir / "Radius_of_Gyration.svg"
            png_o = report_dir / "Radius_of_Gyration.png"
            draw_simple_timeseries(svg_o, png_o, t, rg,
                                  "Time (ns)", "Radius of Gyration (\u00c5)",
                                  color=SID_COLOR_PROTEIN, dpi=args.dpi)
            produced["rg_svg"] = str(svg_o)
            produced["rg_png"] = str(png_o)
            print(f"[report] \u2713 Radius of gyration plot saved: {png_o}")
        except Exception as e:
            print(f"[warning] Could not draw Rg plot: {e}")
    else:
        print("[warning] Skipping Rg plot (gmx gyrate failed).")

    # ── 6. Hydrogen bond count ───────────────────────────────────────────────
    hbnum_xvg = report_dir / "hbnum.xvg"
    if gmx_hbond(gmx_bin, topol, traj, hbnum_xvg,
               args.hbond_g1, args.hbond_g2, cwd=report_dir):
        try:
            t, hb = read_xvg(hbnum_xvg)
            svg_o = report_dir / "Hydrogen_Bonds.svg"
            png_o = report_dir / "Hydrogen_Bonds.png"
            draw_simple_timeseries(svg_o, png_o, t, hb,
                                  "Time (ns)", "Number of H-Bonds",
                                  color=SID_COLOR_LIGAND, dpi=args.dpi)
            produced["hbond_svg"] = str(svg_o)
            produced["hbond_png"] = str(png_o)
            print(f"[report] \u2713 H-bond count plot saved: {png_o}")
        except Exception as e:
            print(f"[warning] Could not draw H-bond plot: {e}")
    else:
        print("[warning] Skipping H-bond plot (gmx hbond failed — "
             "this tool name varies across GROMACS versions).")

    # ── 7. SASA ──────────────────────────────────────────────────────────────
    ndx_file = report_dir / "index.ndx"
    sasa_xvg = report_dir / "sasa_lig.xvg"
    if gmx_make_combined_index(gmx_bin, topol, ndx_file, cwd=report_dir,
                               group_a=args.sasa_protein_group,
                               group_b=args.sasa_ligand_group):
        if gmx_sasa(gmx_bin, topol, traj, sasa_xvg, ndx_file, cwd=report_dir):
            try:
                t, sasa = read_xvg(sasa_xvg)
                if sasa.max() < 500.0:   # nm² → Å² (×100)
                    sasa = sasa * 100.0
                svg_o = report_dir / "SASA.svg"
                png_o = report_dir / "SASA.png"
                draw_simple_timeseries(svg_o, png_o, t, sasa,
                                      "Time (ns)", "SASA (\u00c5\u00b2)",
                                      color="#6fbf73", dpi=args.dpi)
                produced["sasa_svg"] = str(svg_o)
                produced["sasa_png"] = str(png_o)
                print(f"[report] \u2713 SASA plot saved: {png_o}")
            except Exception as e:
                print(f"[warning] Could not draw SASA plot: {e}")
        else:
            print("[warning] Skipping SASA plot (gmx sasa failed).")
    else:
        print("[warning] Skipping SASA plot (gmx make_ndx failed).")

    print("=" * 70)
    print(f"[report] Full MD report complete. {len(produced)//2} figure(s) generated.")
    print("=" * 70 + "\n")
    return produced


def load_ligand_mol2_for_rmsf(mol2_path: str
                              ) -> Optional[Tuple[Chem.Mol, Dict[int, Tuple[float, float]]]]:
    """
    Load a ligand .mol2 for the RMSF structure panel (heavy atoms, 2D coords).
    Handles atoms with unusual valences (e.g. pentavalent N in nitro groups)
    by skipping the valence-check step of sanitization.
    """
    try:
        mol = Chem.MolFromMol2File(mol2_path, removeHs=False, sanitize=False)
        if mol is None:
            return None

        # Sanitize in steps — skip SANITIZE_PROPERTIES (valence check) on failure
        san_ok = False
        for flags in [
            Chem.SanitizeFlags.SANITIZE_ALL,
            # Skip valence check if full sanitize fails (pentavalent N, etc.)
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            # Last resort: only rings + aromaticity
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
            Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        ]:
            try:
                Chem.SanitizeMol(mol, flags)
                san_ok = True
                break
            except Exception:
                continue

        # Remove hydrogens (sanitize=False to preserve the mol even with valence issues)
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
            # Re-sanitize after H removal, skipping valence check
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass

        # Kekulize (optional — skip silently if it fails)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass

        # Compute 2D coordinates
        rdDepictor.SetPreferCoordGen(True)
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        coords = {}
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            coords[i] = (p.x * 1.6, p.y * 1.6)
        return mol, coords
    except Exception as e:
        print(f"[warning] load_ligand_mol2_for_rmsf failed: {e}")
        return None


# =============================================================================
# Main pipeline
# =============================================================================

def run_analysis(args):
    print("=" * 70)
    print(f"[VERSION] SCRIPT_VERSION = {SCRIPT_VERSION}")
    print("=" * 70)
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Topology  : {args.topol}")
    print(f"[info] Trajectory: {args.traj}")
    universe  = mda.Universe(args.topol, args.traj)
    ligand_ag = universe.select_atoms(args.ligand_selection)
    if ligand_ag.n_atoms == 0:
        raise ValueError(f"Ligand '{args.ligand_selection}' → 0 atoms")

    # ── Protein (for ProLIF + ionic)
    protein_ag = universe.select_atoms(args.protein_selection)
    if protein_ag.n_atoms == 0:
        raise ValueError(f"Protein '{args.protein_selection}' → 0 atoms")
    print(f"[info] Ligand atoms : {ligand_ag.n_atoms}")
    print(f"[info] Protein atoms: {protein_ag.n_atoms}")

    # ── Metal ions (separate, for custom metal detector)
    metal_ag = None
    if args.metal_selection:
        metal_ag = universe.select_atoms(args.metal_selection)
        if metal_ag.n_atoms > 0:
            print(f"[info] Metal atoms  : {metal_ag.n_atoms}  "
                  f"resnames={set(metal_ag.resnames)}")
            # Also add to protein_ag so ProLIF can see them
            protein_ag = protein_ag + metal_ag
        else:
            print("[warning] --metal-selection matched 0 atoms")
            metal_ag = None

    # ── Water
    water_ag = None
    if args.water_selection:
        water_ag = universe.select_atoms(args.water_selection)
        if water_ag.n_atoms == 0:
            print("[warning] Water selection → 0 atoms; WaterBridge disabled")
            water_ag = None
        else:
            print(f"[info] Water atoms  : {water_ag.n_atoms}")

    # ── 2-D ligand (no H)
    print("[info] Building 2D ligand (heavy atoms only) …")
    depiction_mol, idx_map = build_depiction_mol(ligand_ag, args.ligand_ref)
    coords  = get_2d_atom_coords(depiction_mol)
    rings   = get_aromatic_rings(depiction_mol)
    centers = ring_centers(rings, coords)
    print(f"[info] Heavy atoms in 2D mol: {depiction_mol.GetNumAtoms()}")

    # ── ProLIF fingerprint
    fp = build_fingerprint(water_ag, args.include_vdw, args.water_order)

    traj_slice = universe.trajectory[args.start:args.stop:args.stride]
    n_frames   = len(traj_slice)
    if n_frames == 0:
        raise ValueError("Trajectory slice has 0 frames")
    print(f"[info] Analysing {n_frames} frames with ProLIF …")

    try:
        fp.run(traj_slice, ligand_ag, protein_ag,
               progress=not args.no_progress)
    except TypeError:
        fp.run(traj_slice, ligand_ag, protein_ag)

    # ── Parse ProLIF results
    print("[info] Parsing ProLIF results …")
    all_occurrences = parse_prolif_occurrences(
        fp, n_frames, depiction_mol, coords, rings, centers, idx_map)

    # ── Custom geometric detectors (re-iterate same trajectory slice)
    traj_slice2 = universe.trajectory[args.start:args.stop:args.stride]

    # Water Bridge (custom — always run if water present)
    if water_ag is not None:
        wb_occs = detect_water_bridges(
            universe, ligand_ag, protein_ag, water_ag, traj_slice2,
            coords, depiction_mol, idx_map,
            cutoff=args.water_cutoff,
        )
        # Remove any ProLIF WaterBridge results to avoid double counting
        all_occurrences = [o for o in all_occurrences
                           if o.group != "WaterBridge"]
        all_occurrences.extend(wb_occs)

    # Ionic (custom — always run)
    traj_slice3 = universe.trajectory[args.start:args.stop:args.stride]
    ion_occs = detect_ionic(
        universe, ligand_ag, protein_ag, traj_slice3,
        coords, depiction_mol, idx_map,
        cutoff=args.ionic_cutoff,
    )
    # Merge: remove ProLIF ionic, add custom
    all_occurrences = [o for o in all_occurrences
                       if o.group != "Ionic"]
    all_occurrences.extend(ion_occs)

    # Metal (custom)
    if metal_ag is not None:
        traj_slice4 = universe.trajectory[args.start:args.stop:args.stride]
        met_occs = detect_metal(
            universe, ligand_ag, metal_ag, traj_slice4,
            coords, depiction_mol, idx_map,
            cutoff=args.metal_cutoff,
        )
        all_occurrences = [o for o in all_occurrences
                           if o.group != "Metal"]
        all_occurrences.extend(met_occs)

    # ── Aggregate
    # Build per-group threshold dict from individual CLI args
    per_group_min = {
        "HydrogenBond": args.min_hbond,
        "WaterBridge":  args.min_water,
        "Hydrophobic":  args.min_hydrophobic,
        "PiStacking":   args.min_pi,
        "PiCation":     args.min_pi,
        "Ionic":        args.min_ionic,
        "Metal":        args.min_metal,
        "Halogen":      args.min_halogen,
        "VdWContact":   args.min_percent,
    }
    interactions = aggregate_occurrences(all_occurrences, n_frames,
                                          args.min_percent,
                                          per_group_min=per_group_min)
    if args.max_interactions > 0:
        interactions = interactions[:args.max_interactions]

    print(f"\n[info] Retained (>={args.min_percent:.1f}%): {len(interactions)}")
    grp_sum: Dict[str, List[str]] = defaultdict(list)
    for it in interactions:
        grp_sum[it.group].append(
            f"{it.residue_name}{it.residue_number}({it.percentage:.0f}%)")
    for g, items in sorted(grp_sum.items()):
        print(f"       {g:22s}: {', '.join(items)}")

    if not interactions:
        print("[warning] Nothing to draw."); return

    # ── Layout + figure
    bbox   = ligand_bbox(coords)
    center = coords.mean(axis=0)
    nodes  = build_residue_nodes(interactions, center, bbox,
                                  min_gap_deg=args.min_gap_deg)

    # Build H2O node positions for water bridges (spread on inner ellipse)
    wb_interactions = [it for it in interactions if it.group == "WaterBridge"]
    water_positions = _build_water_node_positions(
        wb_interactions, center, bbox, nodes)

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    draw_ligand(ax, depiction_mol, coords)

    for it in interactions:
        node = nodes[it.residue_key]
        if it.group == "WaterBridge":
            wpos = water_positions.get(f"{it.residue_key}|{it.anchor.key}")
            if wpos is not None:
                draw_waterbridge(ax, it, node, wpos)
            else:
                draw_direct_interaction(ax, it, node)
        else:
            draw_direct_interaction(ax, it, node)

    for node in nodes.values():
        draw_residue_node(ax, node)

    draw_title(ax, args.title, bbox, nodes)

    xmin, xmax, ymin, ymax = final_limits(coords, nodes)
    groups_present = list(dict.fromkeys(it.group for it in interactions))
    draw_legend(ax, groups_present, xmin, ymin)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.2)

    svg_path  = Path(f"{args.output_prefix}.svg")
    png_path  = Path(f"{args.output_prefix}.png")
    html_path = Path(f"{args.output_prefix}_interactive.html")
    csv_path  = Path(f"{args.output_prefix}_interactions.csv")
    json_path = Path(f"{args.output_prefix}_interactions.json")

    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=args.dpi)
    plt.close(fig)
    export_csv(csv_path, interactions)
    export_json(json_path, interactions)

    # ── Fully interactive HTML (drag nodes + pan/zoom) ────────────────────
    export_interactive_html(
        html_path, depiction_mol, coords, interactions,
        nodes, water_positions, title=args.title or "")

    # ── Protein-Ligand Contacts Histogram (Schrödinger/Desmond style) ─────
    print("[info] Building protein-ligand contacts histogram …")
    fractions, res_info = compute_residue_group_fractions(
        all_occurrences, n_frames)
    hist_svg = Path(f"{args.output_prefix}_contacts_histogram.svg")
    hist_png = Path(f"{args.output_prefix}_contacts_histogram.png")
    hist_title = (f"{args.title} — Contacts Histogram" if args.title
                  else "Protein\u2013Ligand Contacts Histogram")
    hist_ok = draw_contacts_histogram(
        hist_svg, hist_png, fractions, res_info,
        min_fraction=args.hist_min_percent / 100.0,
        title=hist_title,
        width=args.hist_width, height=args.hist_height,
        dpi=args.dpi)

    result_summary = {
        "svg":  str(svg_path),  "png": str(png_path),
        "html": str(html_path), "csv": str(csv_path),
        "json": str(json_path),
        "frames_analyzed":       n_frames,
        "interactions_retained": len(interactions),
    }
    if hist_ok:
        result_summary["contacts_histogram_svg"] = str(hist_svg)
        result_summary["contacts_histogram_png"] = str(hist_png)

    # ── Optional: full automated MD report (RMSD/RMSF/H-bond/Rg/SASA) ───────
    if args.full_report:
        contact_residues = {it.residue_number for it in interactions}
        try:
            report_files = run_full_md_report(args, contact_residues=contact_residues)
            result_summary.update(report_files)
        except Exception as e:
            print(f"[error] Full MD report failed unexpectedly: {e}")

    print("\n[done]", json.dumps(result_summary, indent=2))


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Schrodinger-like 2D ligand interaction diagram (GROMACS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--topol",             required=True)
    p.add_argument("--traj",              required=True)
    p.add_argument("--ligand-selection",  required=True, help='"resname LIG"')
    p.add_argument("--protein-selection", default="protein")
    p.add_argument("--metal-selection",
                   default="resname ZN ZN2 ZNB MG MG2 MN FE FE2 CU NI CO CA",
                   help="Metal ions (added to protein AND used for custom metal detector). "
                        "Set '' to disable.")
    p.add_argument("--water-selection",
                   default="resname SOL HOH WAT TIP3 TIP3P SPC SPCE")
    p.add_argument("--ligand-ref",   default=None,
                   help="SDF/MOL2/PDB for correct bond orders")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--title",         default=None)
    p.add_argument("--start",   type=int,   default=None)
    p.add_argument("--stop",    type=int,   default=None)
    p.add_argument("--stride",  type=int,   default=1)
    p.add_argument("--water-order",      type=int,   default=1)
    p.add_argument("--water-cutoff",     type=float, default=3.5,
                   help="Water bridge O–atom cutoff (Å)")
    p.add_argument("--ionic-cutoff",     type=float, default=4.5,
                   help="Ionic interaction cutoff (Å)")
    p.add_argument("--metal-cutoff",     type=float, default=2.8,
                   help="Metal coordination cutoff (Å)")
    p.add_argument("--min-percent",      type=float, default=4.0,
                   help="Global fallback minimum %% frequency")
    # Per-interaction-type thresholds (higher = fewer, less crowded diagram)
    p.add_argument("--min-hbond",        type=float, default=20.0,
                   help="Min %% for H-Bond")
    p.add_argument("--min-hydrophobic",  type=float, default=25.0,
                   help="Min %% for Hydrophobic")
    p.add_argument("--min-water",        type=float, default=15.0,
                   help="Min %% for Water Bridge")
    p.add_argument("--min-ionic",        type=float, default=15.0,
                   help="Min %% for Ionic / Salt Bridge")
    p.add_argument("--min-metal",        type=float, default=10.0,
                   help="Min %% for Metal coordination")
    p.add_argument("--min-pi",           type=float, default=20.0,
                   help="Min %% for Pi-stacking / Pi-cation")
    p.add_argument("--min-halogen",      type=float, default=15.0,
                   help="Min %% for Halogen bond")
    p.add_argument("--max-interactions", type=int,   default=0)
    p.add_argument("--min-gap-deg",      type=float, default=28.0,
                   help="Min angular gap between residue nodes (°)")
    p.add_argument("--include-vdw",  action="store_true")
    p.add_argument("--dpi",    type=int,   default=600)
    p.add_argument("--width",  type=float, default=18.0)
    p.add_argument("--height", type=float, default=12.0)
    p.add_argument("--no-progress", action="store_true")
    # ── Contacts histogram options ────────────────────────────────────────
    p.add_argument("--hist-min-percent", type=float, default=1.0,
                   help="Min %% of frames in contact to include a residue "
                        "in the contacts histogram")
    p.add_argument("--hist-width",  type=float, default=14.0,
                   help="Contacts histogram base width (inches)")
    p.add_argument("--hist-height", type=float, default=5.0,
                   help="Contacts histogram height (inches)")

    # ── Full automated MD report (RMSD / RMSF / H-bond / Rg / SASA) ────────
    p.add_argument("--full-report", action="store_true",
                   help="Automatically run gmx rms/rmsf/hbond/gyrate/sasa "
                        "and generate Schrodinger/Desmond-style RMSD & RMSF "
                        "figures, reusing the same --topol/--traj/--ligand-ref.")
    p.add_argument("--gmx-bin", default="gmx",
                   help="GROMACS executable name/path")
    p.add_argument("--report-dir", default=None,
                   help="Output directory for the full report "
                        "(default: same folder as --output-prefix)")
    p.add_argument("--secondary-structure", action="store_true", default=True,
                   help="Attempt secondary-structure shading on the protein "
                        "RMSF plot via `gmx dssp` (silently skipped if "
                        "unavailable). Use --no-secondary-structure to disable.")
    p.add_argument("--no-secondary-structure", dest="secondary_structure",
                   action="store_false")
    # gmx rms — protein Cα RMSD: select group twice (fit, then RMSD calc)
    p.add_argument("--rmsd-protein-g1", type=int, default=3,
                   help="gmx rms protein: 1st selection (least-squares fit group)")
    p.add_argument("--rmsd-protein-g2", type=int, default=3,
                   help="gmx rms protein: 2nd selection (RMSD group)")
    # gmx rms — ligand RMSD fit on protein
    p.add_argument("--rmsd-ligand-g1", type=int, default=3,
                   help="gmx rms ligand: 1st selection (fit group, e.g. C-alpha)")
    p.add_argument("--rmsd-ligand-g2", type=int, default=13,
                   help="gmx rms ligand: 2nd selection (RMSD group, e.g. LIG)")
    # gmx rmsf
    p.add_argument("--rmsf-protein-group", type=int, default=4,
                   help="gmx rmsf -res selection for protein (e.g. Backbone)")
    p.add_argument("--rmsf-ligand-group", type=int, default=13,
                   help="gmx rmsf selection for ligand (e.g. LIG)")
    # gmx hbond
    p.add_argument("--hbond-g1", type=int, default=1,
                   help="gmx hbond: 1st selection (e.g. Protein)")
    p.add_argument("--hbond-g2", type=int, default=13,
                   help="gmx hbond: 2nd selection (e.g. LIG)")
    # gmx gyrate
    p.add_argument("--gyrate-group", type=int, default=1,
                   help="gmx gyrate selection (e.g. Protein)")
    # gmx sasa (via make_ndx merge group_a | group_b -> 'Protein_LIG')
    p.add_argument("--sasa-protein-group", type=int, default=1,
                   help="make_ndx group number for Protein (merged for SASA)")
    p.add_argument("--sasa-ligand-group", type=int, default=13,
                   help="make_ndx group number for LIG (merged for SASA)")

    return p


def main():
    run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
