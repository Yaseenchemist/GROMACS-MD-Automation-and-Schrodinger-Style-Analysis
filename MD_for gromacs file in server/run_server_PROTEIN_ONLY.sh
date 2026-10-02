#!/bin/bash
###############################################################################
#                                                                             #
#   run_server_PROTEIN_ONLY.sh                                                #
#   محاكاة GROMACS لـ (بروتين لوحده — بدون أي ليجند) — على سيرفر Aphroditi      #
#   التحضير على الـ login node، والتشغيل على GPU عبر SLURM                      #
#   ملفات mdp تُكتب تلقائياً داخل هذا الكود                                      #
#                                                                             #
#   الاستعمال (على الـ front node):                                            #
#     1) حط ملف البروتين بنفس المجلد وسمّه  protein.pdb  (أو عدّل INPUT_PDB)     #
#     2) شغّل:   bash run_server_PROTEIN_ONLY.sh                                #
#     3) يبني النظام ثم يرسل وظيفة GPU بـ sbatch ويطبع JobID                    #
#                                                                             #
###############################################################################
set -e

# ═══════════════════════════════════════════════════════════════════════════
#   ①  الإعدادات  — عدّل هنا
# ═══════════════════════════════════════════════════════════════════════════
INPUT_PDB="protein.pdb"                                # ← اسم ملف البروتين
FF="charmm27"                                          # ← القوة الحقلية (بالاسم)
WATER="tip3p"                                          # ← نموذج الماء
BOX_DIST="1.0"                                         # ← مسافة الجدار (nm)
SALT_CONC="0.1"                                        # ← تركيز الملح (M)
PROD_NS=100                                            # ← مدة الإنتاج (ns)
TEMP=300                                               # ← الحرارة (K)

GMX_MODULE="GROMACS/2025.3-foss-2025a-CUDA-12.9.1"     # ← تحقّق: module avail GROMACS
SLURM_ACCOUNT="p307"                                   # ← حسابك
SLURM_PARTITION="gpu"                                  # ← تحقّق: sinfo -s
SLURM_GRES="gpu:1"                                     # ← أو gpu:v100:1
SLURM_TIME="24:00:00"
SLURM_CPUS=16

NSTEPS_PROD=$(( PROD_NS * 500000 ))                    # 1 ns = 500000 خطوة

echo "════════════════════════════════════════════════"
echo "  PROTEIN-ONLY MD (SERVER / GPU via SLURM)"
echo "  Input=$INPUT_PDB  FF=$FF  Water=$WATER  Prod=$PROD_NS ns  T=$TEMP K"
echo "════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════
#   ②  فحوصات + تحميل GROMACS
# ═══════════════════════════════════════════════════════════════════════════
[[ -f "$INPUT_PDB" ]] || { echo "ERROR: ما لقيت '$INPUT_PDB'"; exit 1; }
module purge
module load "$GMX_MODULE"

# ═══════════════════════════════════════════════════════════════════════════
#   ③  كتابة ملفات mdp تلقائياً (بروتين لوحده: tc-grps = Protein Water_and_ions)
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
#   ④  بناء النظام على الـ login node
# ═══════════════════════════════════════════════════════════════════════════
echo "--- [1] cleaning input (ATOM+TER only) ---"
grep -E "^(ATOM|TER)" "$INPUT_PDB" > protein_clean.pdb

echo "--- [2] pdb2gmx (FF=$FF, water=$WATER) ---"
gmx pdb2gmx -f protein_clean.pdb -ignh -ff "$FF" -water "$WATER" -o conf.gro -p topol.top

echo "--- [3] box ---"
gmx editconf -f conf.gro -d "$BOX_DIST" -bt triclinic -o box.gro

echo "--- [4] solvate ---"
gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro

echo "--- [5] ions ---"
gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -maxwarn 2 -o ION.tpr
echo "SOL" | gmx genion -s ION.tpr -p topol.top -conc "$SALT_CONC" -neutral -o box_sol_ion.gro

# ═══════════════════════════════════════════════════════════════════════════
#   ⑤  توليد سكربت SLURM (EM → NVT → NPT → MD على الـ GPU) وإرساله
# ═══════════════════════════════════════════════════════════════════════════
echo "--- [6] writing md_gpu.slurm ---"
cat > md_gpu.slurm << SLURM
#!/bin/bash
#SBATCH --job-name=prot_md
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --gres=$SLURM_GRES
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
gmx grompp -f NVT.mdp -c EM.gro  -r EM.gro  -p topol.top -maxwarn 2 -o NVT.tpr
gmx mdrun  -v -deffnm NVT \$GPUFLAGS

echo "### NPT ###"
gmx grompp -f NPT.mdp -c NVT.gro -r NVT.gro -t NVT.cpt -p topol.top -maxwarn 2 -o NPT.tpr
gmx mdrun  -v -deffnm NPT \$GPUFLAGS

echo "### MD ($PROD_NS ns) ###"
gmx grompp -f MD.mdp  -c NPT.gro -t NPT.cpt -p topol.top -maxwarn 2 -o MD.tpr
gmx mdrun  -v -deffnm MD  \$GPUFLAGS -update gpu
echo "### DONE — MD.xtc MD.gro MD.tpr MD.edr MD.log ###"
SLURM

echo "--- [7] submitting ---"
JOBID=$(sbatch md_gpu.slurm | awk '{print $NF}')
echo "════════════════════════════════════════════════"
echo "  JobID = $JOBID"
echo "  المتابعة:  squeue -j $JOBID   |   tail -f md_${JOBID}.out"
echo "════════════════════════════════════════════════"
