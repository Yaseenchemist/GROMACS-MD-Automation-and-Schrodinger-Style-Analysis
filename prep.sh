#!/bin/bash
set -e
echo "=== GROMACS System Preparation ==="

module purge
module load GROMACS/2025.3-foss-2025a-CUDA-12.9.1

# ── 1. توليد طبولوجيا البروتين (CHARMM27) ──────────────────
gmx pdb2gmx -f REC.pdb -ignh << EOF
9
1
EOF

# ── 2. تحويل الليكند إلى .gro ───────────────────────────────
gmx editconf -f LIG.pdb -o LIG.gro

# ── 3. دمج الليكند مع conf.gro ──────────────────────────────
python3 - << 'PYEOF'
with open("LIG.gro") as f:
    lig_lines = f.readlines()[2:-1]

with open("conf.gro", "r") as f:
    conf_lines = f.readlines()

total_atoms = int(conf_lines[1].strip()) + len(lig_lines)
conf_lines[1] = f" {total_atoms}\n"
conf_lines = conf_lines[:-1] + lig_lines + [conf_lines[-1]]

with open("conf.gro", "w") as f:
    f.writelines(conf_lines)
print("conf.gro updated.")
PYEOF

# ── 4. إضافة LIG.itp إلى topol.top ─────────────────────────
python3 - << 'PYEOF'
with open("topol.top") as f:
    lines = f.readlines()

new_lines = []
inserted = False
for line in lines:
    new_lines.append(line)
    if not inserted and "forcefield.itp" in line:
        new_lines.append('; Include ligand topology\n#include "LIG.itp"\n')
        inserted = True

with open("topol.top", "w") as f:
    f.writelines(new_lines)
print("LIG.itp added to topol.top.")
PYEOF

# ── 5. بناء الصندوق وإضافة المذيب ───────────────────────────
gmx editconf -f conf.gro -d 1.0 -bt triclinic -o box.gro
gmx solvate  -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro

# ── 6. إضافة LIG إلى [ molecules ] قبل SOL ──────────────────
python3 - << 'PYEOF'
with open("topol.top") as f:
    lines = f.readlines()

new_lines = []
done = False
for line in lines:
    if not done and line.strip().startswith("SOL"):
        new_lines.append("LIG             1\n")
        done = True
    new_lines.append(line)

with open("topol.top", "w") as f:
    f.writelines(new_lines)
print("LIG added before SOL in [ molecules ].")
PYEOF

# ── 7. إضافة الأيونات ────────────────────────────────────────
gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -maxwarn 2 -o ION.tpr
echo "15" | gmx genion -s ION.tpr -p topol.top -conc 0.1 -neutral -o box_sol_ion.gro

# ── 8. تقليل الطاقة (Energy Minimization) ───────────────────
gmx grompp -f EM.mdp -c box_sol_ion.gro -p topol.top -maxwarn 2 -o EM.tpr
gmx mdrun   -v -deffnm EM -nb gpu

# ── 9. قيود الموضع للليكند ───────────────────────────────────
echo -e "0 & ! a H*\nq" | gmx make_ndx -f LIG.gro -o index_LIG.ndx
echo "3" | gmx genrestr -f LIG.gro -n index_LIG.ndx -o posre_LIG.itp \
            -fc 1000 1000 1000

python3 - << 'PYEOF'
with open("topol.top") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "; Include Position restraint file" in line:
        new_lines.append('#ifdef POSRES\n#include "posre.itp"\n#endif\n')
        new_lines.append('; Ligand position restraints\n')
        new_lines.append('#ifdef POSRES\n#include "posre_LIG.itp"\n#endif\n')

with open("topol.top", "w") as f:
    f.writelines(new_lines)
print("Position restraints added.")
PYEOF

# ── 10. ملف الفهرس الكامل ────────────────────────────────────
echo -e "1 | 13\nq" | gmx make_ndx -f EM.gro -o index.ndx

echo "=== Preparation Complete — Ready for MD ==="
