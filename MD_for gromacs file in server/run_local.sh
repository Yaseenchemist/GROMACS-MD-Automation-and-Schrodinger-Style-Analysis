#!/bin/bash
###############################################################################
#  run_local.sh  —  Protein-Ligand MD على اللاب توب (CPU فقط، بدون python)
#  الـ NVIDIA معطّلة، فنُجبر CPU (GMX_DISABLE_GPU_DETECTION).
#  المطلوب:  complex.pdb + ions.mdp EM.mdp NVT.mdp NPT.mdp MD.mdp
#  التشغيل:  bash run_local.sh
###############################################################################
set -e

GMXRC="/usr/local/gromacs/bin/GMXRC"
NT=$(nproc)
SP_HOST="https://www.swissparam.ch:8443"
SP_OPTS="approach=both&c27"
FF="charmm27"
WATER="tip3p"
export GMX_DISABLE_GPU_DETECTION=1

echo "=== WORKDIR: $(pwd)   |   CPU threads: $NT ==="
for f in complex.pdb ions.mdp EM.mdp NVT.mdp NPT.mdp MD.mdp; do
    [[ -f "$f" ]] || { echo "ERROR: مفقود $f"; exit 1; }
done
source "$GMXRC"
command -v obabel >/dev/null || echo "WARNING: obabel غير متوفر — لازم LIG.pdb + LIG.itp محضّرين مسبقاً."

###############################################################################
#  1) Ligand + SwissParam
###############################################################################
echo "=== [1/6] Ligand + SwissParam ==="
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
#  2) بناء النظام (بدون python)
###############################################################################
echo "=== [2/6] System preparation (FF=$FF, water=$WATER) ==="

gmx pdb2gmx -f REC.pdb -ignh -ff "$FF" -water "$WATER"
gmx editconf -f LIG.pdb -o LIG.gro

nlig=$(sed -n '2p' LIG.gro  | tr -d '[:space:]')
nconf=$(sed -n '2p' conf.gro | tr -d '[:space:]')
newcount=$((nconf + nlig))
{ head -n 1 conf.gro; echo " $newcount"; sed '1,2d;$d' conf.gro; sed '1,2d;$d' LIG.gro; tail -n 1 conf.gro; } > conf.new
mv conf.new conf.gro
echo "  conf.gro merged ($nconf + $nlig = $newcount atoms)."

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

gmx editconf -f conf.gro -d 1.0 -bt triclinic -o box.gro
gmx solvate  -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro

awk -v name="$LIGNAME" '
  /\[[[:space:]]*molecules[[:space:]]*\]/ { inmol=1; print; next }
  inmol==1 && ($1==name || $1=="LIG") && $2 ~ /^[0-9]+$/ { next }
  inmol==1 && $1=="SOL" && !written { print name "             1"; written=1 }
  { print }' topol.top > topol.tmp && mv topol.tmp topol.top

gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -maxwarn 2 -o ION.tpr
echo "SOL" | gmx genion -s ION.tpr -p topol.top -conc 0.1 -neutral -o box_sol_ion.gro

echo -e "0 & ! a H*\nq" | gmx make_ndx -f LIG.gro -o index_LIG.ndx
echo "3" | gmx genrestr -f LIG.gro -n index_LIG.ndx -o posre_LIG.itp -fc 1000 1000 1000

gmx select -s box_sol_ion.gro -on index.ndx -select \
'"Protein_LIG" not resname SOL NA CL; "Water_and_ions" resname SOL NA CL; "LIG" not resname SOL NA CL and not group "Protein"'
echo ">>> index.ndx جاهز."

###############################################################################
#  3) EM  4) NVT  5) NPT  6) MD  (CPU)
###############################################################################
echo "=== [3/6] EM (CPU) ==="
gmx grompp -f EM.mdp -c box_sol_ion.gro -p topol.top -maxwarn 2 -o EM.tpr
gmx mdrun  -v -deffnm EM -nb cpu -ntmpi 1 -ntomp "$NT"

echo "=== [4/6] NVT (CPU) ==="
gmx grompp -f NVT.mdp -c EM.gro -r EM.gro -p topol.top -n index.ndx -maxwarn 2 -o NVT.tpr
gmx mdrun  -v -deffnm NVT -nb cpu -ntmpi 1 -ntomp "$NT"

echo "=== [5/6] NPT (CPU) ==="
gmx grompp -f NPT.mdp -c NVT.gro -r NVT.gro -t NVT.cpt -p topol.top -n index.ndx -maxwarn 2 -o NPT.tpr
gmx mdrun  -v -deffnm NPT -nb cpu -ntmpi 1 -ntomp "$NT"

echo "=== [6/6] MD (CPU) ==="
gmx grompp -f MD.mdp -c NPT.gro -t NPT.cpt -p topol.top -n index.ndx -maxwarn 2 -o MD.tpr
gmx mdrun  -v -deffnm MD -nb cpu -ntmpi 1 -ntomp "$NT"

echo "=== انتهى run_local.sh — المخرجات: MD.xtc / MD.gro / MD.edr / MD.log ==="
