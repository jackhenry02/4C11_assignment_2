**Overall**
We took the coursework from skeleton code to a near-submission state for both problems, plus tuning infrastructure, tracking, documentation, and a full report draft in LaTeX.

**Problem 1: PINN**
- Started from the PINN notebook skeleton and made the code device-safe, including `mps`/`cuda`/`cpu` selection, dtype handling, and safe plotting via `detach().cpu().numpy()`.
- Added explicit precision logic so the notebooks could use fast float32/MPS for short tests but fall back to CPU/float64 for reliable final runs.
- Added training diagnostics:
  - total loss vs epochs
  - decomposed loss-component plots
  - saved figures for all major plots
- Reworked part (e) so it is controlled by `USE_DATA_PART_E` instead of commenting blocks in and out.
- Added a fixed random seed and made the selected 50 data points reproducible.
- Added lightweight but robust experiment tracking:
  - per-run folders
  - saved configs
  - JSON tracking files
  - loss histories
  - checkpoints
  - latest/best model saves
  - resume support
- Fixed best-model logic so `best.pt` actually corresponds to the lowest recorded loss, and made final plots load the best checkpoint instead of just using the last in-memory weights.
- Added final FEM-based validation:
  - displacement metrics
  - FEM vs PINN displacement contour plots
  - displacement error plots
- Added residual diagnostics:
  - equilibrium and constitutive residual histograms
  - residual spatial maps
  - residual summary JSON
- Added FEM stress comparison by reconstructing element stresses from the FEM displacement field using the same CST post-processing as the MATLAB reference, then comparing those with PINN stresses.
- Identified and fixed the key formulation bug: the shear term had to use the engineering shear convention consistently everywhere to match the FEM code.
- Built [optuna_pinn_v4.py](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/optuna_pinn_v4.py) with:
  - SQLite-backed Optuna studies
  - dashboard compatibility
  - saved study plots
  - baseline and part (e) study modes
  - rerun-best mode for float32 vs float64 comparisons
- Ran Optuna-style tuning workflows and used the rerun comparisons to choose final configurations rather than trusting short float32 runs blindly.
- Created final long-run notebooks:
  - [PINN_final.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/PINN_final.ipynb)
  - [PINN_data_final.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/PINN_data_final.ipynb)
- Final technical conclusion for Problem 1:
  - physics-only PINN converged in residual terms but matched FEM poorly
  - adding 50 displacement measurements transformed the result and made the PINN genuinely accurate

**Problem 2: Darcy operator learning**
- Re-read the coursework brief and inspected the CNN/FNO skeleton notebooks before designing anything.
- Chose to implement three notebooks:
  - [DARCY_CNN_simple.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/DARCY_CNN_simple.ipynb)
  - [DARCY_CNN_UNet.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/DARCY_CNN_UNet.ipynb)
  - [DARCY_FNO_v2_cpu_check.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/DARCY_FNO_v2_cpu_check.ipynb)
- Kept them close to the original skeleton structure while adding a top config cell, device handling, validation split, grid-search toggles, retrain-on-full-train mode, saved figures, and saved metrics.
- Implemented:
  - a strong classical CNN baseline with pooling and batch normalization
  - a U-Net with matched encoder/decoder depth and skip connections
  - a skeleton-led custom FNO with completed spectral layers, MLP blocks, repeated Fourier blocks, and device-safe FFT/grid handling
- Added manual grid search with internal validation, then retraining on the full training set for final runs.
- Added per-run experiment tracking for all three Darcy notebooks so every run is saved rather than overwritten.
- Added final coursework-required outputs:
  - train/test loss curves
  - truth/prediction contour plots
  - test-sample comparison plots
  - final test error statistic at the bottom of each notebook
- Added [DARCY_truth_plots.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/DARCY_truth_plots.ipynb) to inspect the true Darcy solution fields directly.
- Updated the final Darcy notebook hyperparameters to the best settings found from grid search.
- Diagnosed the key Problem 2 numerical issue:
  - the FNO on MPS produced FFT warnings and poor results
  - rerunning the same FNO on CPU gave dramatically better performance
- Final technical conclusion for Problem 2:
  - simple CNN was a strong baseline
  - U-Net improved on it
  - CPU FNO was best overall
  - backend reliability was essential to the final conclusion

**Documentation and write-up**
- Added heavy explanatory comments to the main five notebooks and the Optuna script without changing their logic.
- Wrote:
  - [hyperparameters.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/hyperparameters.md)
  - [problem_1_improvements.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/problem_1_improvements.md)
  - [problem_2_improvements.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/problem_2_improvements.md)
  - [report_plan.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/report_plan.md)
- Wrote the full report draft in:
  - [coursework_report.tex](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/coursework_report.tex)
  - with [coursework_report.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/coursework_report.md) as the earlier source draft

**Net result**
You now have:
- final PINN notebooks for no-data and data-assisted runs
- three finished Darcy model notebooks plus a truth-plot notebook
- tuning/tracking infrastructure for both problems
- saved figures and metrics for the canonical runs
- improvement notes for both problems
- a long-form LaTeX report draft that already reflects the final technical story of the coursework