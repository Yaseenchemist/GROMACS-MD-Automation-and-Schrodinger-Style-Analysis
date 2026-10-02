#!/bin/bash
###############################################################################
#                                                                             #
#   run_local_PROTEIN_ONLY.sh                                                 #
#   محاكاة GROMACS لـ (بروتين لوحده — بدون أي ليجند) — تشغيل محلي على اللاب توب #
#   CPU فقط (الـ NVIDIA معطّلة) — الملفات mdp تُكتب تلقائياً داخل هذا الكود      #
#                                                                             #
#   الاستعمال:                                                                 #
#     1) حط ملف البروتين بنفس المجلد وسمّه  protein.pdb  (أو عدّل INPUT_PDB)     #
#     2) شغّل:   bash run_local_PROTEIN_ONLY.sh                                 #
#                                                                             #
###############################################################################
set -e

# ═══════════════════════════════════════════════════════════════════════════
#   ①  الإعدادات  — عدّل هنا فقط
# ═══════════════════════════════════════════════════════════════════════════
INPUT_PDB="protein.pdb"                 # ← اسم ملف البروتين المدخل
GMXRC="/usr/local/gromacs/bin/GMXRC"    # ← مسار GROMACS المحلي (2023)
FF="charmm27"                           # ← القوة الحقلية (بالاسم، مو بالرقم)
WATER="tip3p"                           # ← نموذج الماء
BOX_DIST="1.0"                          # ← مسافة البروتين لجدار الصندوق (nm)
SALT_CONC="0.1"                         # ← تركيز الملح (M)
PROD_NS=100                             # ← مدة الإنتاج (نانو ثانية)
TEMP=300                                # ← درجة الحرارة (K)
NT=$(nproc)                             # كل أنوية المعالج

export GMX_DISABLE_GPU_DETECTION=1      # يجبر CPU (GPU معطّلة)
NSTEPS_PROD=$(( PROD_NS * 500000 ))     # 1 ns = 500000 خطوة عند dt=0.002

echo "════════════════════════════════════════════════"
echo "  PROTEIN-ONLY MD (LOCAL / CPU)"
echo "  Input=$INPUT_PDB  FF=$FF  Water=$WATER"
echo "  Production=$PROD_NS ns ($NSTEPS_PROD steps)  T=$TEMP K"
echo "  Threads=$NT"
echo "════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════
#   ②  فحوصات + تحميل GROMACS
# ═══════════════════════════════════════════════════════════════════════════
[[ -f "$INPUT_PDB" ]] || { echo "ERROR: ما لقيت ملف البروتين '$INPUT_PDB'"; exit 1; }
[[ -f "$GMXRC" ]] && source "$GMXRC"
command -v gmx >/dev/null || { echo "ERROR: gmx مو موجود — عدّل GMXRC"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════════
#   ③  كتابة ملفات mdp تلقائياً  (بروتين لوحده: tc-grps = Protein Water_and_ions)
# ═══════════════════════════════════════════════════════════════════════════
echo "--- writing mdp files ---"

cat > ions.mdp << 'EOF'
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
nstlist         = 1
cutoff-scheme   = Verlet
coulombtype     = cutoff
rcoulomb        = 1.0
rvdw            = 1.0
rlist           = 1.0
pbc             = xyz
EOF

cat > EM.mdp << 'EOF'
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
nstlist         = 1
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
vdwtype         = cutoff
vdw-modifier    = force-switch
rvdw-switch     = 1.0
rvdw            = 1.2
rlist           = 1.2
pbc             = xyz
DispCorr        = no
EOF

cat > NVT.mdp << EOF
define                  = -DPOSRES
integrator              = md
nsteps                  = 50000
dt                      = 0.002
nstenergy               = 500
nstlog                  = 500
nstxout-compressed      = 500
continuation            = no
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw-switch             = 1.0
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
pme_order               = 4
fourierspacing          = 0.16
tcoupl                  = V-rescale
tc-grps                 = Protein Water_and_ions
tau_t                   = 0.1   0.1
ref_t                   = $TEMP  $TEMP
pcoupl                  = no
pbc                     = xyz
DispCorr                = no
gen_vel                 = yes
gen_temp                = $TEMP
gen_seed                = -1
EOF

cat > NPT.mdp << EOF
define                  = -DPOSRES
integrator              = md
nsteps                  = 50000
dt                      = 0.002
nstenergy               = 500
nstlog                  = 500
nstxout-compressed      = 500
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw-switch             = 1.0
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
pme_order               = 4
fourierspacing          = 0.16
tcoupl                  = V-rescale
tc-grps                 = Protein Water_and_ions
tau_t                   = 0.1   0.1
ref_t                   = $TEMP  $TEMP
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
pbc                     = xyz
DispCorr                = no
gen_vel                 = no
EOF

cat > MD.mdp << EOF
integrator              = md
nsteps                  = $NSTEPS_PROD
dt                      = 0.002
nstenergy               = 5000
nstlog                  = 5000
nstxout-compressed      = 5000
compressed-x-grps       = System
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw-switch             = 1.0
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
pme_order               = 4
fourierspacing          = 0.16
tcoupl                  = V-rescale
tc-grps                 = Protein Water_and_ions
tau_t                   = 0.1   0.1
ref_t                   = $TEMP  $TEMP
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 5.0
ref_p                   = 1.0
compressibility         = 4.5e-5
pbc                     = xyz
DispCorr                = no
gen_vel                 = no
EOF

# ═══════════════════════════════════════════════════════════════════════════
#   ④  بناء النظام  (تنظيف → طوبولوجيا → صندوق → مذيب → أيونات)
# ═══════════════════════════════════════════════════════════════════════════
echo "--- [1] cleaning input (keep ATOM+TER only, drop any HETATM/water/ion) ---"
grep -E "^(ATOM|TER)" "$INPUT_PDB" > protein_clean.pdb

echo "--- [2] pdb2gmx (FF=$FF, water=$WATER) ---"
gmx pdb2gmx -f protein_clean.pdb -ignh -ff "$FF" -water "$WATER" -o conf.gro -p topol.top

echo "--- [3] box (triclinic, ${BOX_DIST} nm) ---"
gmx editconf -f conf.gro -d "$BOX_DIST" -bt triclinic -o box.gro

echo "--- [4] solvate ---"
gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro

echo "--- [5] ions (${SALT_CONC} M + neutral) ---"
gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -maxwarn 2 -o ION.tpr
echo "SOL" | gmx genion -s ION.tpr -p topol.top -conc "$SALT_CONC" -neutral -o box_sol_ion.gro

# ═══════════════════════════════════════════════════════════════════════════
#   ⑤  التشغيل (CPU):  EM → NVT → NPT → MD
#       ملاحظة: tc-grps = Protein Water_and_ions مجموعات افتراضية،
#       فما نحتاج ملف index — grompp يولّدها تلقائياً.
# ═══════════════════════════════════════════════════════════════════════════
echo "--- [6] Energy Minimization ---"
gmx grompp -f EM.mdp -c box_sol_ion.gro -p topol.top -maxwarn 2 -o EM.tpr
gmx mdrun -v -deffnm EM -nb cpu -ntmpi 1 -ntomp "$NT"

echo "--- [7] NVT (100 ps, restrained) ---"
gmx grompp -f NVT.mdp -c EM.gro -r EM.gro -p topol.top -maxwarn 2 -o NVT.tpr
gmx mdrun -v -deffnm NVT -nb cpu -ntmpi 1 -ntomp "$NT"

echo "--- [8] NPT (100 ps, restrained) ---"
gmx grompp -f NPT.mdp -c NVT.gro -r NVT.gro -t NVT.cpt -p topol.top -maxwarn 2 -o NPT.tpr
gmx mdrun -v -deffnm NPT -nb cpu -ntmpi 1 -ntomp "$NT"

echo "--- [9] Production MD ($PROD_NS ns) ---"
gmx grompp -f MD.mdp -c NPT.gro -t NPT.cpt -p topol.top -maxwarn 2 -o MD.tpr
gmx mdrun -v -deffnm MD -nb cpu -ntmpi 1 -ntomp "$NT"

echo "════════════════════════════════════════════════"
echo "  DONE — المخرجات: MD.xtc  MD.gro  MD.tpr  MD.edr  MD.log"
echo "════════════════════════════════════════════════"
