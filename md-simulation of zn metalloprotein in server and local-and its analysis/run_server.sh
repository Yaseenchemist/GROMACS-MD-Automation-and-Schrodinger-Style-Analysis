#!/bin/bash
###############################################################################
#  run_server.sh — Protein-Ligand(+Zn Metalloprotein) MD على Aphrodite
#  (The Cyprus Institute) — يدعم CHARMM36 + بروتين عادي أو ميتالوبروتين يحتوي
#  زنك (Zn2+). الكود يعمل تلقائيًا في الحالتين (وجود/عدم وجود الزنك) وكذلك في
#  حالة وجود/عدم وجود ليكاند عضوي منفصل.
#  يُشغّل على الـ front node.  (بدون python — awk/sed فقط)
#    (أ) SwissParam + تحضير النظام  -> login node
#    (ب) mdrun (EM/NVT/NPT/MD)      -> وظيفة GPU عبر SLURM
#  المطلوب:  complex.pdb + ions.mdp EM.mdp NVT.mdp NPT.mdp MD.mdp
#            (complex.pdb: ATOM records للبروتين + HETATM لأي ليكاند عضوي
#             + HETATM لأيون الزنك (resname/atom = ZN) إن كان بروتينك ميتالوبروتين)
#  اختياري:  مجلد charmm36*.ff جاهز بجانب السكربت (إن لم يوجد سيحاول تنزيله).
#  التشغيل:  bash run_server.sh
###############################################################################
set -e

# ─────────────── الإعدادات ───────────────
GMX_MODULE="GROMACS/2025.3-foss-2025a-CUDA-12.9.1"
SLURM_ACCOUNT="p307"
SLURM_PARTITION="gpu"          # ← تحقّق:  sinfo
SLURM_GRES="gpu:1"             # أو gpu:v100:1
SLURM_TIME="24:00:00"
SLURM_NODES=1
SLURM_NTASKS=1
SLURM_CPUS=16
SP_HOST="https://www.swissparam.ch:8443"
SP_OPTS="approach=both&c27"
WATER="tip3p"
# رابط تنزيل حزمة CHARMM36 الجاهزة لـ GROMACS في حال عدم وجودها محليًا — من
# الصفحة الرسمية لمختبر MacKerell (المصدر الأصلي المعترف به لهذا المنفذ):
#   http://mackerell.umaryland.edu/charmm_ff.shtml#gromacs
# هذا هو أحدث إصدار متوفر هناك وقت كتابة هذا السكربط. ملاحظة: صفحة التنزيل قد
# تتطلب الموافقة على اتفاقية ترخيص عبر المتصفح، فلو فشل curl تلقائيًا (رسالة
# خطأ أدناه) نزّل الملف يدويًا من الرابط أعلاه وارفعه إلى الـ front node بجانب
# السكربت باسم مجلد "charmm36*.ff"، أو حدّث المتغيّر التالي إلى إصدار أحدث إن وُجد.
CHARMM36_URL="https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-feb2026_cgenff-5.0.ff.tgz"

echo "=== WORKDIR: $(pwd) ==="
cp complex.pdb complex_clean.pdb
ls -lh complex.pdb complex_clean.pdb
for f in complex.pdb ions.mdp EM.mdp NVT.mdp NPT.mdp MD.mdp; do
    [[ -f "$f" ]] || { echo "ERROR: مفقود $f"; exit 1; }
done
module purge
module load "$GMX_MODULE"
command -v obabel >/dev/null || echo "WARNING: obabel غير متوفر — لازم LIG.pdb + LIG.itp محضّرين مسبقاً."

###############################################################################
#  0) تجهيز قوة الحقل CHARMM36 (تُنزَّل تلقائيًا لو غير موجودة محليًا)
###############################################################################
echo "=== [0/4] CHARMM36 force field ==="
FF_DIR=$(find . -maxdepth 1 -iname "charmm36*.ff" -type d | sort | head -n 1)
if [[ -z "$FF_DIR" ]]; then
    echo ">>> ماكو مجلد charmm36*.ff محليًا — محاولة تنزيله..."
    if [[ -n "$CHARMM36_URL" ]] && curl -fsSL --max-time 60 "$CHARMM36_URL" -o charmm36.ff.tgz; then
        tar -xzf charmm36.ff.tgz
        rm -f charmm36.ff.tgz
        FF_DIR=$(find . -maxdepth 1 -iname "charmm36*.ff" -type d | sort | head -n 1)
    fi
fi
if [[ -z "$FF_DIR" ]]; then
    cat <<EOF
ERROR: تعذّر إيجاد/تنزيل قوة الحقل CHARMM36.
  الحل: نزّل حزمة "charmm36-xxxx.ff" (منفذ CHARMM36 الخاص بـ GROMACS، مثلاً من
  mackerell.umaryland.edu أو أي مصدر موثوق)، وفكّها هنا بجوار هذا السكربت على
  الـ front node بحيث يظهر مجلد اسمه charmm36*.ff في نفس مجلد العمل، ثم أعد
  التشغيل.
EOF
    exit 1
fi
FF="${FF_DIR#./}"; FF="${FF%.ff}"
echo "  FF = $FF   (من المجلد: $FF_DIR)"

# اكتشاف أسماء الأيونات الفعلية داخل ملفات هذه النسخة من CHARMM36 (تختلف بين
# النسخ: بعضها SOD/CLA على طريقة CHARMM، وبعضها NA/CL؛ والزنك أحيانًا داخل
# ملف اسمه ions.rtp وأحيانًا ضمن solvent.rtp أو metals.rtp) بدل افتراضها يدويًا.
# لذلك نفتّش داخل *كل* ملفات .rtp لقوة الحقل، لا اسم ملف واحد ثابت.
RTP_FILES=$(find "$FF_DIR" -maxdepth 1 -iname "*.rtp")
find_ion_resname() {
    local cand rtp
    for cand in "$@"; do
        for rtp in $RTP_FILES; do
            grep -qE "^\[[[:space:]]*${cand}[[:space:]]*\]" "$rtp" && { echo "$cand"; return 0; }
        done
    done
    return 1
}
PNAME=$(find_ion_resname SOD NA "NA+") || PNAME="SOD"
NNAME=$(find_ion_resname CLA CL "CL-") || NNAME="CLA"
ZNAME=$(find_ion_resname ZN2 ZN "ZN+2") || ZNAME="ZN2"
echo "  أسماء الأيونات في CHARMM36 هذه: كاتيون=$PNAME  أنيون=$NNAME  زنك=$ZNAME"
echo "  (تحقّق منها يدويًا داخل ملفات .rtp في $FF_DIR إذا اختلفت تسمية إصدارك)"

###############################################################################
#  1) فصل الليكاند العضوي عن أيون الزنك (إن وُجد) + obabel + SwissParam
#     (تُتخطّى مرحلة الليكاند/الزنك المُحضَّرة مسبقاً إن وُجدت)
###############################################################################
echo "=== [1/4] Ligand / Zn separation + SwissParam ==="
grep "^ATOM"   complex.pdb > REC.pdb
grep "^HETATM" complex.pdb > HET_ALL.pdb || true
: > LIG1.pdb
: > ZN_raw.pdb

if [[ -s HET_ALL.pdb ]]; then
    awk '
      {
        elem  = substr($0,77,2); gsub(/[ \t]/,"",elem)
        resn  = substr($0,18,3); gsub(/[ \t]/,"",resn)
        atomn = substr($0,13,4); gsub(/[ \t]/,"",atomn)
        isZn = (elem=="ZN" || resn=="ZN" || resn=="ZN2" || atomn=="ZN")
        if (isZn) print > "ZN_raw.pdb"; else print > "LIG1.pdb"
      }' HET_ALL.pdb
fi

if [[ -s ZN_raw.pdb ]]; then
    HAS_ZN=1
    n_zn=$(wc -l < ZN_raw.pdb)
    echo ">>> عُثر على $n_zn أيون/ذرّة زنك — سيُعامَل كجزء من المستقبِل (ميتالوبروتين)."
    awk -v newname="$ZNAME" '{ printf "%s%3s%s\n", substr($0,1,17), newname, substr($0,21) }' ZN_raw.pdb > ZN.pdb
else
    HAS_ZN=0
    echo ">>> ماكو زنك داخل complex.pdb — تشغيل عادي كبروتين غير ميتالوبروتين."
fi

if [[ -s LIG1.pdb || -f LIG.itp ]]; then
    HAS_LIG=1
else
    HAS_LIG=0
    echo ">>> ماكو ليكاند عضوي منفصل داخل complex.pdb — تخطّي مرحلة obabel/SwissParam."
fi

if [[ "$HAS_LIG" == 1 ]]; then
    if [[ -f LIG.pdb && -f LIG.itp ]]; then
        echo ">>> LIG.pdb و LIG.itp موجودان — تخطّي obabel."
    else
        obabel LIG1.pdb -O LIG.pdb  --minimize --ff MMFF94 --steps 1000 --crit 1e-6
        obabel LIG.pdb  -O LIG.mol2 --title "LIG"
    fi

    if [[ -f LIG.itp ]]; then
        echo ">>> LIG.itp موجود — تخطّي SwissParam."
    else
        response=$(curl -s -F "myMol2=@LIG.mol2" "$SP_HOST/startparam?$SP_OPTS")
        session_number=$(echo "$response" | grep -o '[0-9]\+' | head -n 1)
        echo "Session: $session_number"
        [[ -n "$session_number" ]] || { echo "ERROR: SwissParam ما رجّع session"; exit 1; }
        while true; do
            result=$(curl -s "$SP_HOST/checksession?sessionNumber=$session_number")
            echo "  $result"
            echo "$result" | grep -qi "finished" && break
            sleep 120
        done
        curl -s "$SP_HOST/retrievesession?sessionNumber=$session_number" -o results.tar.gz
        tar -xzf results.tar.gz
        itp_file=$(find . -name "*.itp" ! -name "posre*" | head -n 1)
        par_file=$(find . \( -name "*.par" -o -name "*.prm" \) 2>/dev/null | head -n 1)
        [[ -z "$itp_file" ]] && { echo "ERROR: ماكو .itp — فشل SwissParam"; exit 1; }
        cp "$itp_file" LIG.itp
        [[ -n "$par_file" ]] && cp "$par_file" LIG.par
    fi
    echo "  LIG.itp جاهز."
fi

###############################################################################
#  2) بناء النظام (login node، بدون python)
###############################################################################
echo "=== [2/4] System preparation (FF=$FF, water=$WATER) ==="

# 2.0 دمج الزنك (إن وُجد) داخل ملف المستقبِل مع TER قبله وبعده حتى يتعامل
#     pdb2gmx معه كجزيء أيوني منفصل (نموذج غير-رابطي/nonbonded) باستخدام
#     تعريف الزنك الجاهز داخل قوة حقل CHARMM36 نفسها — بدون أي itp يدوي.
if [[ "$HAS_ZN" == 1 ]]; then
    { cat REC.pdb; echo "TER"; cat ZN.pdb; echo "TER"; } > REC_full.pdb
else
    cp REC.pdb REC_full.pdb
fi

# -chainsep ter: نفرض فصل السلاسل عند TER فقط (صريح وحتمي) لتفادي أي سؤال
# تفاعلي من pdb2gmx بخصوص فصل السلاسل نفسه.
#
# -ter: مهم جدًا مع بعض إصدارات CHARMM36 الحديثة (مثل charmm36-feb2026) —
# بدون -ter يختار pdb2gmx تلقائيًا نمط النهاية الطرفية (N-/C-terminus)، وهذا
# التلقائي معروف أنه يفشل أحيانًا مع الميثيونين الطرفي N (MET1) في بعض نسخ
# CHARMM36 برسالة "atom C1 not found in building block 1MET" (خطأ موثّق في
# منتدى GROMACS الرسمي لمنفذ CHARMM36). فبإضافة -ter نجبر pdb2gmx يسألك
# صراحة عن نوع كل نهاية طرفية بدل التخمين الآلي الخاطئ.
# ⚠ بما إنها تفاعلية: نفّذ هذه الخطوة (على الـ front node) من محطة طرفية
# عادية لا بالخلفية، وعند ظهور القائمة اختر الخيار القياسي المشحون: عادة
# "NH3+" (أو ما يعادلها كخيار 0) للبداية N، و"COO-" (أو خيار 0) للنهاية C.
# لو ظهر سؤال عن سلسلة الزنك (عادة لا يظهر لأنه أيون مفرد بلا نهايات)،
# اختر "None"/0. مرحلة GPU اللاحقة (mdrun عبر SLURM) غير تفاعلية كالمعتاد.
gmx pdb2gmx -f REC_full.pdb -ignh -ff "$FF" -water "$WATER" -chainsep ter -ter

if [[ "$HAS_LIG" == 1 ]]; then
    # 2.2 الليكند -> gro
    gmx editconf -f LIG.pdb -o LIG.gro

    # 2.3 دمج الليكند مع conf.gro  (sed/bash)
    nlig=$(sed -n '2p' LIG.gro  | tr -d '[:space:]')
    nconf=$(sed -n '2p' conf.gro | tr -d '[:space:]')
    newcount=$((nconf + nlig))
    { head -n 1 conf.gro; echo " $newcount"; sed '1,2d;$d' conf.gro; sed '1,2d;$d' LIG.gro; tail -n 1 conf.gro; } > conf.new
    mv conf.new conf.gro
    echo "  conf.gro merged ($nconf + $nlig = $newcount atoms)."

    # 2.4 اسم moleculetype الحقيقي + إدراج LIG.par(لو لازم)+LIG.itp+posre بعد forcefield.itp
    LIGNAME=$(awk 'BEGIN{f=0} /\[[[:space:]]*moleculetype[[:space:]]*\]/{f=1;next} f==1{ if($0 ~ /^[[:space:]]*;/ || $0 ~ /^[[:space:]]*$/) next; print $1; exit }' LIG.itp)
    [[ -n "$LIGNAME" ]] || { echo "ERROR: ماكو [ moleculetype ] داخل LIG.itp"; exit 1; }
    echo "  moleculetype = $LIGNAME"
    if grep -qiE '\[[[:space:]]*atomtypes[[:space:]]*\]' LIG.itp; then HAS_AT=1; else HAS_AT=0; fi
    HASPAR=$([ -f LIG.par ] && echo 1 || echo 0)

    grep -vE '#include "(LIG\.itp|LIG\.par|posre_LIG\.itp)"' topol.top > topol.tmp && mv topol.tmp topol.top
    awk -v hasat="$HAS_AT" -v haspar="$HASPAR" '
      { print }
      !done && /forcefield\.itp/ {
        print "; Include ligand topology"
        if (hasat==0 && haspar==1) print "#include \"LIG.par\""
        print "#include \"LIG.itp\""
        print "; Ligand position restraints"
        print "#ifdef POSRES"; print "#include \"posre_LIG.itp\""; print "#endif"
        done=1
      }' topol.top > topol.tmp && mv topol.tmp topol.top
fi

# 2.5 الصندوق + المذيب
gmx editconf -f conf.gro -d 1.0 -bt triclinic -o box.gro
gmx solvate  -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro

if [[ "$HAS_LIG" == 1 ]]; then
    # 2.6 '<name> 1' قبل SOL في [ molecules ]
    awk -v name="$LIGNAME" '
      /\[[[:space:]]*molecules[[:space:]]*\]/ { inmol=1; print; next }
      inmol==1 && ($1==name || $1=="LIG") && $2 ~ /^[0-9]+$/ { next }
      inmol==1 && $1=="SOL" && !written { print name "             1"; written=1 }
      { print }' topol.top > topol.tmp && mv topol.tmp topol.top
fi
# ملاحظة: جزيء الزنك (إن وُجد) يضيفه pdb2gmx نفسه تلقائيًا داخل [ molecules ]
# ضمن قسم منفصل (Ion_chain_..)، فلا حاجة لأي تعديل يدوي إضافي هنا.

# 2.7 الأيونات (بأسماء الكاتيون/الأنيون المكتشَفة فعليًا من CHARMM36)
gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -maxwarn 2 -o ION.tpr
echo "SOL" | gmx genion -s ION.tpr -p topol.top -pname "$PNAME" -nname "$NNAME" -conc 0.1 -neutral -o box_sol_ion.gro

if [[ "$HAS_LIG" == 1 ]]; then
    # 2.8 قيود موضع الليكند
    echo -e "0 & ! a H*\nq" | gmx make_ndx -f LIG.gro -o index_LIG.ndx
    echo "3" | gmx genrestr -f LIG.gro -n index_LIG.ndx -o posre_LIG.itp -fc 1000 1000 1000
fi

# 2.9 index.ndx بالاسم (نستثني اسمي الأيونات الفعليين لا NA/CL الثابتين، ونُبقي
#     الزنك ضمن مجموعة "Protein_LIG" لأنه جزء بنيوي من المستقبِل وليس ملحًا)
gmx select -s box_sol_ion.gro -on index.ndx -select \
"\"Protein_LIG\" not resname SOL $PNAME $NNAME; \"Water_and_ions\" resname SOL $PNAME $NNAME; \"LIG\" not resname SOL $PNAME $NNAME and not group \"Protein\""
echo ">>> index.ndx جاهز (لو أيوناتك مختلفة، عدّل قائمة resname أعلاه)."
if [[ "$HAS_ZN" == 1 ]]; then
    echo ">>> ملاحظة ميتالوبروتين: الزنك مُمثَّل بنموذج غير-رابطي (كاتيون +2) عبر"
    echo "    قوة حقل CHARMM36 مباشرة. لمحاكاة إنتاجية طويلة على موقع تناسق"
    echo "    رباعي/خماسي حسّاس، يُفضَّل إضافة قيود مسافة اختيارية بين الزنك"
    echo "    والذرات المُنسِّقة (His Nδ1/Nε2, Cys Sγ, Asp/Glu O) للحفاظ على"
    echo "    الهندسة، ويمكن مراجعتها يدويًا في topol.top عند الحاجة."
fi

###############################################################################
#  3) توليد سكربت SLURM لتشغيل الـ GPU
###############################################################################
echo "=== [3/4] توليد md_gpu.slurm ==="
cat > md_gpu.slurm << SLURM
#!/bin/bash
#SBATCH --job-name=pl_md
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --gres=$SLURM_GRES
#SBATCH --nodes=$SLURM_NODES
#SBATCH --ntasks=$SLURM_NTASKS
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --output=md_%j.out
#SBATCH --error=md_%j.err
set -e
module purge
module load $GMX_MODULE
cd "\$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
GPUFLAGS="-nb gpu -pme gpu -bonded gpu -ntmpi 1 -ntomp \$SLURM_CPUS_PER_TASK"

echo "### EM ###"
gmx grompp -f EM.mdp  -c box_sol_ion.gro -p topol.top -maxwarn 2 -o EM.tpr
gmx mdrun  -v -deffnm EM -nb gpu -ntmpi 1 -ntomp \$SLURM_CPUS_PER_TASK

echo "### NVT ###"
gmx grompp -f NVT.mdp -c EM.gro  -r EM.gro  -p topol.top -n index.ndx -maxwarn 2 -o NVT.tpr
gmx mdrun  -v -deffnm NVT \$GPUFLAGS

echo "### NPT ###"
gmx grompp -f NPT.mdp -c NVT.gro -r NVT.gro -t NVT.cpt -p topol.top -n index.ndx -maxwarn 2 -o NPT.tpr
gmx mdrun  -v -deffnm NPT \$GPUFLAGS

echo "### MD 100 ns ###"
gmx grompp -f MD.mdp  -c NPT.gro -t NPT.cpt -p topol.top -n index.ndx -maxwarn 2 -o MD.tpr
gmx mdrun  -v -deffnm MD  \$GPUFLAGS -update gpu
echo "### DONE ###"
SLURM

echo "=== [4/4] إرسال الوظيفة ==="
JOBID=$(sbatch md_gpu.slurm | awk '{print $NF}')
echo ">>> JobID = $JOBID   |   المتابعة: squeue -j $JOBID ; tail -f md_${JOBID}.out"
echo "=== انتهى run_server.sh ==="
