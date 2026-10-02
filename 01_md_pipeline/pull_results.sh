#!/bin/bash
###############################################################################
#  pull_results.sh  —  تنزيل مخرجات الـ MD من Aphroditi إلى اللاب توب
#  يُشغّل محلياً (على اللاب توب)، من داخل المجلد اللي تريد تنزّل فيه.
#  التشغيل:            bash pull_results.sh
#     أو الأساسيات فقط: bash pull_results.sh essentials
###############################################################################
set -e

# ─────────────────── الإعدادات (عدّل REMOTE_DIR) ────────────────────────────
REMOTE_USER="YOUR_USERNAME"
REMOTE_HOST="HPC_LOGIN_NODE"
REMOTE_DIR="/path/to/your_project/gromacs_project/RUN"   # ← مجلد التشغيل على السيرفر
LOCAL_DIR="$(pwd)/results_from_server"

mkdir -p "$LOCAL_DIR"
MODE="${1:-full}"

if [[ "$MODE" == "essentials" ]]; then
    echo "=== تنزيل الملفات الأساسية فقط ==="
    rsync -avP \
      --include='MD.xtc' --include='MD.tpr' --include='MD.gro' \
      --include='MD.edr' --include='MD.log' --include='MD.cpt' \
      --include='NPT.gro' --include='topol.top' --include='index.ndx' \
      --include='LIG.itp' --include='LIG.par' --include='*.itp' \
      --include='complex.pdb' \
      --exclude='*' \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
else
    echo "=== تنزيل مجلد التشغيل بالكامل (باستثناء ملفات backup) ==="
    rsync -avP \
      --exclude='#*#' --exclude='step*.pdb' --exclude='*.trr' \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
fi

echo "=== تم التنزيل إلى:  $LOCAL_DIR ==="
