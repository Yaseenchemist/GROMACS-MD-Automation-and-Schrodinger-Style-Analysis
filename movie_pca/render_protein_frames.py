#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 render_protein_frames.py   —   يشتغل داخل PyMOL (headless / بدون شاشة)
==============================================================================
 يرندر إطارات البروتين+الليكند من الـ trajectory كصور PNG، مع لف 360°.

 GENERAL: يشتغل لأي بروتين + ligand، ومع/بدون Zn أو أي معدن.

 مخرجات:  frames_protein/prot_XXXX.png

 يُستدعى من run_all.sh هكذا:
   pymol -cq render_protein_frames.py -- MD.gro MD_center.xtc LIG 10 white

 الوسائط (بعد --):
   1) topology       (MD.gro / .pdb)
   2) trajectory     (MD_center.xtc)
   3) ligand resname (LIG)          — اسم الـ ligand
   4) stride         (10)           — كل كم إطار
   5) bg color       (white/black)  — لون الخلفية
==============================================================================
"""
import sys
import os
from pymol import cmd

# ── قراءة الوسائط بعد "--" ──────────────────────────────────────────────────
argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
else:
    args = argv[1:]

top       = args[0] if len(args) > 0 else "MD.gro"
traj      = args[1] if len(args) > 1 else "MD_center.xtc"
lig_resn  = args[2] if len(args) > 2 else "LIG"
stride    = int(args[3]) if len(args) > 3 else 10
bg        = args[4] if len(args) > 4 else "white"

OUTDIR = "frames_protein"
os.makedirs(OUTDIR, exist_ok=True)

print(f"[PyMOL] topology   = {top}")
print(f"[PyMOL] trajectory = {traj}")
print(f"[PyMOL] ligand     = {lig_resn}")
print(f"[PyMOL] stride     = {stride}")
print(f"[PyMOL] background = {bg}")

# ── إعدادات المشهد ───────────────────────────────────────────────────────────
cmd.reinitialize()
cmd.bg_color(bg)
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 1)          # حواف واضحة
cmd.set("cartoon_transparency", 0.0)
cmd.set("ray_shadows", 0)             # أسرع، أنظف
cmd.set("depth_cue", 0)
cmd.set("orthoscopic", 1)

# ── تحميل البنية + المسار ────────────────────────────────────────────────────
cmd.load(top, "MD")
cmd.load_traj(traj, "MD", state=1, interval=stride)
n_states = cmd.count_states("MD")
print(f"[PyMOL] loaded states (after stride) = {n_states}")

# ── التمثيل العام (general) ──────────────────────────────────────────────────
cmd.hide("everything", "MD")

# البروتين: cartoon rainbow (N→C spectrum) — general لأي بروتين
cmd.show("cartoon", "MD and polymer.protein")
cmd.spectrum("count", "rainbow", "MD and polymer.protein and name CA")

# الليكند: ball-and-stick (لو موجود)
lig_sel = f"MD and resn {lig_resn}"
if cmd.count_atoms(lig_sel) > 0:
    cmd.show("sticks", lig_sel)
    cmd.set("stick_radius", 0.18, lig_sel)
    cmd.color("grey90", f"{lig_sel} and elem C")
    cmd.util.cnc(lig_sel)             # ألوان CPK للعناصر غير الكربون
    print(f"[PyMOL] ligand '{lig_resn}' shown: {cmd.count_atoms(lig_sel)} atoms")
else:
    print(f"[PyMOL] ⚠️  ligand '{lig_resn}' not found — skipping.")

# المعادن (Zn وغيره) كـ spheres — general (يكتشف تلقائياً)
metal_sel = "MD and (elem Zn+Mg+Ca+Fe+Mn+Na+K+Cu+Ni+Co)"
if cmd.count_atoms(metal_sel) > 0:
    cmd.show("spheres", metal_sel)
    cmd.set("sphere_scale", 0.5, metal_sel)
    cmd.util.cnc(metal_sel)
    print(f"[PyMOL] metals shown: {cmd.count_atoms(metal_sel)} atoms")

# residues حول الليكند (binding site) كـ lines خفيفة
if cmd.count_atoms(lig_sel) > 0:
    cmd.select("pocket", f"byres (MD and polymer.protein within 5 of ({lig_sel}))")
    cmd.show("lines", "pocket")
    cmd.set("line_width", 1.0)

# ── الكاميرا: توسيط وضبط ─────────────────────────────────────────────────────
cmd.orient("MD and polymer.protein")
cmd.zoom("MD and polymer.protein", buffer=5)
cmd.center("MD and polymer.protein")

# ── الرندر: كل state مع لف الكاميرا ─────────────────────────────────────────
W, H = 900, 900
total_rot = 360.0                     # لفة كاملة عبر الفيديو
rot_per_state = total_rot / max(1, n_states)

print(f"[PyMOL] rendering {n_states} frames ({W}x{H}) with 360° rotation ...")
for i in range(1, n_states + 1):
    cmd.set("state", i)
    # لف تدريجي حول المحور y
    cmd.turn("y", rot_per_state)
    out = os.path.join(OUTDIR, f"prot_{i:04d}.png")
    cmd.ray(W, H)
    cmd.png(out, dpi=150)
    if i % 10 == 0 or i == n_states:
        print(f"   frame {i}/{n_states}  →  {out}")

print(f"[PyMOL] ✅ done. {n_states} PNGs in {OUTDIR}/")
