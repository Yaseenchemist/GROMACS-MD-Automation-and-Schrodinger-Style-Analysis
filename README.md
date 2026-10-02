# GROMACS-MD-Automation-and-Schrodinger-Style-Analysis

**One-command GROMACS MD (laptop or SLURM GPU) for protein-ligand and apo systems, plus ligand RMSF, PCA/FEL and PCA+protein movie tools.**

Build and run a protein-ligand (or apo protein) MD simulation with a **single script**, on a laptop (CPU) or on an HPC GPU through
SLURM, then analyse it: ligand RMSF drawn on the 2D structure, PCA / free-energy landscape (FEL) and a combined protein + PCA/FEL movie.

> بالعربي: محاكاة ديناميكية جزيئية كاملة بسكربت واحد (لاب توب أو سيرفر GPU) مع أدوات تحليل: RMSF للليكاند، PCA/FEL، وفيديو يربط حركة البروتين بالـ PCA.

**Related repositories**
- Zinc metalloproteins: [`Zinc-Metalloprotein-MD-CHARMM36-Pipeline`](https://github.com/Yaseenchemist/Zinc-Metalloprotein-MD-CHARMM36-Pipeline)
- 2D interaction diagrams: [`MD-2D-Interaction-Diagram-Schrodinger-Style`](https://github.com/Yaseenchemist/MD-2D-Interaction-Diagram-Schrodinger-Style)

## Structure

| Folder | What it does |
|---|---|
| `01_md_pipeline/` | `run_local.sh` / `run_server.sh` (protein-ligand), `*_PROTEIN_ONLY.sh` (apo protein), mdp files (EM, ions, NVT, NPT, MD), `pull_results.sh`, and the `RUN_ON_SERVER.md` step-by-step guide. `example/complex.pdb` is a sample input. Ligand topology via SwissParam, CHARMM force field, GROMACS 2025.x. |
| `03_analysis/ligand_rmsf/` | Ligand RMSF drawn on the ligand's 2D structure (RDKit) from a GROMACS `.xvg`. |
| `03_analysis/pca_fel/` | `PCA_FEL_ANALYZER.py` (single system or holo vs apo comparison: PCA, cosine content, convergence, FEL), a binding-pocket variant, an Arabic PCA/FEL interpretation guide and a summary of the binding-pocket modifications. |
| `03_analysis/movie_pca/` | `run_all.sh` -> PyMOL frames + FEL 3D / variance / cross-correlation panels -> one MP4. |
| `docs/` | Protein preparation checklist (build the complex from the original crystal protein + docked ligand) and a VMD water-free visualisation reference. |

## Quick start

```bash
bash 01_md_pipeline/run_local.sh      # laptop (CPU)
bash 01_md_pipeline/run_server.sh     # HPC (SLURM + GPU)
```
See the script headers and `docs/` for every option.

## Notes

- Trajectories (`.xtc`), topologies (`.tpr`) and other heavy outputs are intentionally not versioned.
- Cluster usernames/hosts/project IDs are placeholders (`YOUR_USERNAME`, `HPC_LOGIN_NODE`, `YOUR_PROJECT_ID`) - set your own.
- Requires GROMACS, MDAnalysis, RDKit, matplotlib, numpy, scipy; PyMOL + ffmpeg for the movie.


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
- `03_analysis/ligand_rmsf/rmsf_for_lig_shape_like_schrodinger.py`
- `03_analysis/movie_pca/INSTRUCTIONS_AR.md`
- `03_analysis/movie_pca/make_combined_movie.py`
- `03_analysis/movie_pca/render_protein_frames.py`
- `03_analysis/movie_pca/run_all.sh`
- `03_analysis/pca_fel/Binding_Pocket_Analysis_Modifications_Summary_ENGLISH.pdf`
- `03_analysis/pca_fel/PCA_FEL_ANALYZER.py`
- `03_analysis/pca_fel/PCA_FEL_Terms_Interpretation_Guide_AR.docx`
- `03_analysis/pca_fel/pca_fel_pro_binding_pocket.py`
- `docs/Protein_Preparation_Before_MD_AR.md`
- `docs/VMD_GROMACS_Water_Free_Visualization_Reference.pdf`

## Author

Yaseen Saleem Hamdoon - Pharmaceutical chemist & computational drug designer, Lecturer in Organic Pharmaceutical Chemistry, Al-Kitab University (Kirkuk, Iraq).

## License

MIT - see `LICENSE`. Third-party data/tools keep their own licenses.
