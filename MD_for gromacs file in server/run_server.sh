#!/bin/bash
###############################################################################
#  run_server.sh  —  Protein-Ligand MD على Aphroditi (The Cyprus Institute)
#  يُشغّل على الـ front node.  (بدون python — awk/sed فقط)
#    (أ) SwissParam + تحضير النظام  -> login node
#    (ب) mdrun (EM/NVT/NPT/MD)      -> وظيفة GPU عبر SLURM
#  المطلوب:  complex.pdb + ions.mdp EM.mdp NVT.mdp NPT.mdp MD.mdp
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
FF="charmm27"                  # ← بالاسم (مو بالرقم) لتفادي اختلاف ترتيب القائمة
WATER="tip3p"

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
#  1) استخراج الليكند + obabel + SwissParam (تُتخطّى لو محضّرة مسبقاً)
###############################################################################
echo "=== [1/4] Ligand + SwissParam ==="
grep "^ATOM"   complex.pdb > REC.pdb
grep "^HETATM" complex.pdb > LIG1.pdb
[[ -s LIG1.pdb ]] || { echo "ERROR: ماكو HETATM (ليكند) داخل complex.pdb"; exit 1; }

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

###############################################################################
#  2) بناء النظام (login node، بدون python)
###############################################################################
echo "=== [2/4] System preparation (FF=$FF, water=$WATER) ==="

# 2.1 طوبولوجيا البروتين — القوة الحقلية بالاسم
gmx pdb2gmx -f REC.pdb -ignh -ff "$FF" -water "$WATER"

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

# 2.5 الصندوق + المذيب
gmx editconf -f conf.gro -d 1.0 -bt triclinic -o box.gro
gmx solvate  -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro

# 2.6 '<name> 1' قبل SOL في [ molecules ]
awk -v name="$LIGNAME" '
  /\[[[:space:]]*molecules[[:space:]]*\]/ { inmol=1; print; next }
  inmol==1 && ($1==name || $1=="LIG") && $2 ~ /^[0-9]+$/ { next }
  inmol==1 && $1=="SOL" && !written { print name "             1"; written=1 }
  { print }' topol.top > topol.tmp && mv topol.tmp topol.top

# 2.7 الأيونات
gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -maxwarn 2 -o ION.tpr
echo "SOL" | gmx genion -s ION.tpr -p topol.top -conc 0.1 -neutral -o box_sol_ion.gro

# 2.8 قيود موضع الليكند
echo -e "0 & ! a H*\nq" | gmx make_ndx -f LIG.gro -o index_LIG.ndx
echo "3" | gmx genrestr -f LIG.gro -n index_LIG.ndx -o posre_LIG.itp -fc 1000 1000 1000

# 2.9 index.ndx بالاسم
gmx select -s box_sol_ion.gro -on index.ndx -select \
'"Protein_LIG" not resname SOL NA CL; "Water_and_ions" resname SOL NA CL; "LIG" not resname SOL NA CL and not group "Protein"'
echo ">>> index.ndx جاهز (لو أيوناتك مو NA/CL، عدّل قائمة resname)."

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
