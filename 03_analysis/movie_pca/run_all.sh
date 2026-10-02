#!/usr/bin/env bash
# ============================================================================
#  run_all.sh  —  يبني فيديو البروتين+الليكند مع FEL/PCA بأمر واحد
# ============================================================================
#  الاستخدام:
#    bash run_all.sh MD.gro MD_center.xtc LIG 10 white "name CA" final_movie.mp4
#
#  الوسائط (كلها اختيارية، لها قيم افتراضية):
#    1  topology       (MD.gro)
#    2  trajectory     (MD_center.xtc)
#    3  ligand resname (LIG)
#    4  stride         (10)      ← كل كم إطار (زِدها لو بطيء)
#    5  bg color       (white)   ← خلفية صور البروتين
#    6  selection      (name CA) ← لتحليل PCA
#    7  output mp4      (final_movie.mp4)
# ============================================================================
set -e

TOP=${1:-MD.gro}
TRAJ=${2:-MD_center.xtc}
LIG=${3:-LIG}
STRIDE=${4:-10}
BG=${5:-white}
SEL=${6:-name CA}
OUT=${7:-final_movie.mp4}

echo "=============================================="
echo " Protein-Ligand PCA Movie Builder"
echo "=============================================="
echo " topology   : $TOP"
echo " trajectory : $TRAJ"
echo " ligand     : $LIG"
echo " stride     : $STRIDE"
echo " background : $BG"
echo " selection  : $SEL"
echo " output     : $OUT"
echo "=============================================="

# ── فحص المتطلبات ────────────────────────────────────────────────────────────
command -v pymol  >/dev/null 2>&1 || { echo "❌ pymol غير موجود بالبيئة"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "❌ ffmpeg غير موجود — نصّبه: conda install -c conda-forge ffmpeg -y"; exit 1; }
python -c "import mdtraj, sklearn, scipy, matplotlib" 2>/dev/null || { echo "❌ حزم Python ناقصة"; exit 1; }

# ── المرحلة 1: رندر البروتين بـ PyMOL (headless) ───────────────────────────
echo ""
echo "▶️  [1/2] PyMOL rendering (بروتين+ليكند، لف 360°) ..."
LIBGL_ALWAYS_SOFTWARE=1 pymol -cq render_protein_frames.py -- "$TOP" "$TRAJ" "$LIG" "$STRIDE" "$BG"

# ── المرحلة 2: دمج اللوحات + الفيديو ───────────────────────────────────────
echo ""
echo "▶️  [2/2] بناء الفيديو النهائي (FEL/PCA/variance + دمج) ..."
python make_combined_movie.py \
    --top "$TOP" --traj "$TRAJ" \
    --frames-dir frames_protein \
    --stride "$STRIDE" --sel "$SEL" --out "$OUT"

echo ""
echo "=============================================="
echo " ✅ خلص! الفيديو النهائي: $OUT"
echo "=============================================="
