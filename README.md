# GROMACS-MD-Automation-and-Schrodinger-Style-Analysis

**One-command GROMACS MD (laptop or SLURM GPU), Zn-metalloprotein support, plus Schrodinger-style 2D interaction diagrams, ligand RMSF, PCA/FEL and PCA movies.**

A complete MD workflow for protein-ligand and **zinc-metalloprotein** systems: build + run on a laptop (CPU) or an
HPC GPU through SLURM with a single script, then analyse the trajectory with publication-style tools that mimic
the Schrodinger Maestro look (2D interaction diagrams, ligand RMSF on the 2D structure, PCA / free-energy landscape,
and a combined protein + PCA/FEL rotating movie).

> بالعربي: محاكاة ديناميكية جزيئية كاملة بسكربت واحد (لاب توب أو سيرفر GPU)، تدعم بروتينات الزنك، مع أدوات تحليل بشكل Schrodinger.

## Structure

| Folder | What it does |
|---|---|
| `01_md_pipeline/` | `run_local.sh` / `run_server.sh` (protein-ligand), `*_PROTEIN_ONLY.sh` (apo protein), mdp files (EM, ions, NVT, NPT, MD), `pull_results.sh`, and the `RUN_ON_SERVER.md` step-by-step guide. `example/complex.pdb` is a sample input. Ligand topology via SwissParam, CHARMM force field, GROMACS 2025.x. |
| `02_zinc_metalloprotein_md/` | Same pipeline variants for **Zn2+ metalloproteins** with CHARMM36 (auto-detects zinc / ligand presence) + `Zinc_Metal_Tutorial.md` on metal-coordination analysis. |
| `03_analysis/interaction_diagram/` | `gmx_2d_interaction_diagram.py` - Schrodinger-style 2D protein-ligand interaction diagram from a GROMACS trajectory (ProLIF + custom geometry: H-bond, hydrophobic, pi-stacking, pi-cation, halogen, water bridge, ionic, **metal coordination**; binding-site mode). Latest version at top level, version history in `archive/`. |
| `03_analysis/ligand_rmsf/` | Ligand RMSF drawn on the ligand's 2D structure (RDKit) from a GROMACS `.xvg`. |
| `03_analysis/pca_fel/` | `PCA_FEL_ANALYZER.py` (single system or holo vs apo comparison: PCA, cosine content, convergence, FEL), binding-pocket variants, plus an Arabic PCA/FEL interpretation guide. |
| `03_analysis/movie_pca/` | `run_all.sh` -> PyMOL frames + FEL 3D/variance/cross-correlation panels -> one MP4. |
| `docs/` | Protein preparation checklist (build the complex from the original crystal protein + docked ligand), full analysis manual, VMD water-free visualisation reference. |

## Quick start

```bash
# laptop
bash 01_md_pipeline/run_local.sh
# HPC (SLURM + GPU)
bash 01_md_pipeline/run_server.sh
# 2D interaction diagram
python3 03_analysis/interaction_diagram/gmx_2d_interaction_diagram.py --topol MD.tpr --traj MD_center.xtc \
    --ligand-selection "resname LIG" --protein-selection "protein"
```
See the script headers and `docs/` for every option.

## Notes

- Trajectories (`.xtc`), topologies (`.tpr`) and other heavy outputs are intentionally not versioned.
- Cluster usernames/hosts/project IDs are placeholders (`YOUR_USERNAME`, `HPC_LOGIN_NODE`, `YOUR_PROJECT_ID`) - set your own.
- The CHARMM36 force-field folder is not redistributed here: the scripts download it from the MacKerell lab, or place a `charmm36*.ff` folder next to the script.
- Requires GROMACS, MDAnalysis, ProLIF, RDKit, matplotlib, numpy, scipy; PyMOL + ffmpeg for the movie.


## Repository map

- `01_md_pipeline/EM.mdp`
- `01_md_pipeline/MD.mdp`
- `01_md_pipeline/NPT.mdp`
- `01_md_pipeline/NVT.mdp`
- `01_md_pipeline/RUN_ON_SERVER.md`
- `01_md_pipeline/example/complex.pdb`
- `01_md_pipeline/ions.mdp`
- `01_md_pipeline/manual_prep_example/prep.sh`
- `01_md_pipeline/pull_results.sh`
- `01_md_pipeline/run_local.sh`
- `01_md_pipeline/run_local_PROTEIN_ONLY.sh`
- `01_md_pipeline/run_server.sh`
- `01_md_pipeline/run_server_PROTEIN_ONLY.sh`
- `02_zinc_metalloprotein_md/Zinc_Metal_Tutorial.md`
- `02_zinc_metalloprotein_md/example/LIG.mol2`
- `02_zinc_metalloprotein_md/run_local.sh`
- `02_zinc_metalloprotein_md/run_server.sh`
- `03_analysis/interaction_diagram/archive/gmx_2d_interaction_diagram_binding_site_v42.py`
- `03_analysis/interaction_diagram/archive/gmx_2d_interaction_diagram_v17.py`
- `03_analysis/interaction_diagram/archive/gmx_2d_interaction_diagram_v35.py`
- `03_analysis/interaction_diagram/archive/gmx_2d_interaction_diagram_v38c.py`
- `03_analysis/interaction_diagram/archive/gmx_2d_interaction_diagram_v5_early.py`
- `03_analysis/interaction_diagram/gmx_2d_interaction_diagram.py`
- `03_analysis/ligand_rmsf/rmsf_for_lig_shape_like_schrodinger.py`
- `03_analysis/movie_pca/INSTRUCTIONS_AR.md`
- `03_analysis/movie_pca/make_combined_movie.py`
- `03_analysis/movie_pca/render_protein_frames.py`
- `03_analysis/movie_pca/run_all.sh`
- `03_analysis/pca_fel/Binding_Pocket_Analysis_Modifications_Summary_ENGLISH.pdf`
- `03_analysis/pca_fel/PCA_FEL_ANALYZER.py`
- `03_analysis/pca_fel/PCA_FEL_Terms_Interpretation_Guide_AR.docx`
- `03_analysis/pca_fel/pca_fel_pro_binding_pocket.py`
- `03_analysis/pca_fel/pca_fel_pro_binding_pocket_zn_model.py`
- `docs/MD_Analysis_Manual_AR.docx`
- `docs/Protein_Preparation_Before_MD_AR.md`
- `docs/VMD_GROMACS_Water_Free_Visualization_Reference.pdf`

## Author

Yaseen Saleem Hamdoon - Pharmaceutical chemist & computational drug designer, Lecturer in Organic Pharmaceutical Chemistry, Al-Kitab University (Kirkuk, Iraq).

## License

MIT - see `LICENSE`. Third-party data/tools keep their own licenses.
