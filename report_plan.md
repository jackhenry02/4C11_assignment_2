# Final Report Plan

## Summary
- Write an **exhaustive report**, targeting **at least 12–15 pages** in the main body, with no artificial cap if the added detail is relevant and analytical rather than repetitive.
- Organise it **primarily by problem**, with explicit subsection labels matching the coursework prompts so the marker can immediately see where each required item is answered.
- Keep the report technically honest and analytical:
  - include weak or surprising results when they help explain the methods
  - interpret every major figure
  - explain *why* results occurred, not just *what* happened
- Use a **small overall introduction and conclusion**, and also a **short intro and conclusion for each problem** to improve readability.
- When deciding whether to include a plot, include it if you have relevant analysis to attach to it.

## Canonical Evidence Set
- **Problem 1**
  - Physics-only PINN: `problem1_20260315_165315_full_nodata_seed1234`
  - Data-assisted PINN: `problem1_20260315_203703_full_data_seed1234`
  - Optuna/tuning evidence:
    - `problem_1_experiments/optuna_studies/pinn_v4_baseline/...`
    - `problem_1_experiments/optuna_studies/pinn_v4_part_e/...`
- **Problem 2**
  - Simple CNN main result: `20260316_114818_single_final` (`300` epochs, `mps`)
  - U-Net main result: `20260316_115014_single_final` (`400` epochs, `mps`)
  - FNO main result: `20260316_124531_single_final` (`500` epochs, `cpu`)
  - FNO comparison run to discuss:
    - `20260316_122017_single_final` (`500` epochs, `mps`)
- Do **not** treat the later `100`-epoch simple CNN CPU run as canonical.

## Report Structure

### 1. Short Overall Introduction
- One brief section, not a formal abstract.
- State:
  - the coursework has two distinct aims:
    - solving a PDE with a PINN
    - learning a PDE solution operator from data
  - the report will compare methods using:
    - training behaviour
    - quantitative metrics
    - graphical evidence
    - tuning methodology
- Add one sentence that the work also examines **practical numerical issues**, especially:
  - precision/device effects in PINNs
  - backend sensitivity in the FNO

---

## Problem 1: PINN for the Plate-With-Hole Problem

### 2. Problem 1 Introduction
- Short orientation paragraph:
  - goal of the PINN
  - why the problem is difficult
  - why the FEM reference is needed for external validation

### 3. Governing Equations and Boundary Conditions
- Directly answer coursework `(a)`.
- Include:
  - equilibrium PDEs under plane stress
  - constitutive relation
  - stress/strain definitions
  - all boundary conditions:
    - left symmetry
    - bottom symmetry
    - right traction
    - top traction
    - traction-free hole boundary
- Explain briefly why only a quarter plate is modelled.

### 4. PINN Loss Function
- Directly answer coursework `(b)`.
- Write the full composite loss mathematically.
- Break it into:
  - equilibrium loss
  - constitutive loss in the interior
  - constitutive consistency on the boundary
  - boundary-condition loss
  - optional data loss from part `(e)`
- Explain the role of each term and why minimizing a low residual does not necessarily guarantee FEM agreement.

### 5. PINN Architecture and Training Setup
- Directly answer coursework `(c)`.
- Describe:
  - displacement network
  - stress network
  - why the two-network setup was used
  - optimizer and learning-rate schedule choices over the project lifecycle
  - checkpointing / best-model saving logic
- Explain the key implementation correction:
  - the engineering shear convention had to be made consistent with the FEM reference
- Mention that final reported models were run on CPU in double precision for reliability.

### 6. Hyperparameter Tuning Strategy
- This should be a substantive section, not a brief note.
- Explain the tuning workflow:
  - short exploratory runs
  - Optuna studies for baseline and part `(e)`
  - float32/MPS proxy runs for speed
  - float32 vs float64 rerun comparisons on the best trials
  - final full runs in float64/CPU
- Explicitly state the **search spaces**:
  - `learning_rate`: `1e-4` to `3e-3` log scale
  - `disp_width`: `200, 300, 400`
  - `stress_width`: `300, 400, 500`
  - `scheduler`: `none` or `step`
  - `lr_gamma`: `0.30` to `0.70`
  - `lr_step_size`: `2000, 4000, 6000`
  - `w_bc`: `0.5, 1.0, 2.0, 5.0`
  - `w_cons_bc`: `0.5, 1.0, 2.0`
  - `w_data`: `10` to `1000` log scale for part `(e)`
- Explain why **MPS float32** was used during tuning:
  - approximately **3x faster** than float64/CPU-scale runs
  - acceptable for screening
  - not trusted blindly for final conclusions
- Include the **most relevant tuning plots**:
  - `optimization_history.png`
  - `param_importances.png`
  - `parallel_coordinate.png`
  - `rerun_float32_vs_float64.png`
- Explain the meaning of the Optuna plots rather than treating them as decorative:
  - which hyperparameters mattered most
  - whether scheduler use helped or not
  - what the rerun comparison showed about precision/device effects
- Use the rerun comparison summaries to explain:
  - baseline performance remained poor even after tuning
  - part `(e)` became dramatically better
  - float32 vs float64 ranking can change materially

### 7. Physics-Only PINN Results
- Cover coursework `(c)`, `(d)`, and `(f)` using the no-data model.
- Include:
  - training loss curve
  - loss-component curve
  - `sigma_11` plot
  - FEM-vs-PINN displacement figure
  - FEM-vs-PINN stress figure
  - residual distribution or spatial residual map
- Use the quantitative metrics explicitly:
  - displacement relative `L2` about `0.844`
  - stress relative errors large, especially `sigma22` and `sigma12`
- Main analytical message:
  - the model learned to satisfy its **internal physics objective**
  - but this did **not** translate into good agreement with the FEM truth
- Discuss visually:
  - whether the stress concentration region is captured
  - where displacement and stress errors are largest
  - what the residual plots reveal about localized failure modes

### 8. Data-Assisted PINN Results
- Cover coursework `(e)` and `(f)` using the 50 measurement points.
- Include the same figure set as above.
- Use the key metrics explicitly:
  - displacement relative `L2` about `0.0237`
  - `sigma11` relative `L2` about `0.070`
  - clear improvement in all major comparisons
- Main analytical message:
  - adding 50 displacement measurements transformed the problem
  - the total loss was higher than the physics-only case, but external accuracy was far better
- Explain why:
  - the data anchors the displacement field
  - the optimization target becomes better aligned with FEM accuracy
  - stress recovery improves because the displacement field is far better constrained
- Analyse the figures:
  - how the error field shrinks
  - whether the high-stress band near the hole is now captured correctly
  - which components still remain hardest, especially `sigma22`

### 9. Problem 1 Discussion
- Compare the no-data and data-assisted PINNs directly.
- Explicitly highlight the key lesson:
  - **low PINN residual is not sufficient**
  - **small supervised data can dramatically improve solution quality**
- Mention the precision/device lesson:
  - float32/MPS was useful for tuning
  - float64/CPU was preferred for final trustworthy runs
- Use the float32-vs-float64 rerun comparison figure in this discussion if it reads better here than in the tuning section.

### 10. Problem 1 Improvements
- Add a short but serious subsection based on [problem_1_improvements.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/problem_1_improvements.md).
- Include the most relevant future improvements:
  - tune on longer budgets closer to the final `50k–100k` runs
  - use a composite Optuna objective, not just displacement error
  - tune loss weights more systematically
  - test Adam-to-LBFGS refinement
  - improve sampling and validation strategy
  - strengthen checkpoint/best-model evaluation logic further
- Keep this section reflective rather than defensive.

### 11. Problem 1 Conclusion
- One short conclusion paragraph:
  - physics-only PINN struggled despite low residual loss
  - data-assisted PINN achieved strong FEM agreement
  - this is the defensible final result for Problem 1

---

## Problem 2: Learning the Darcy Solution Operator

### 12. Problem 2 Introduction
- Short orientation paragraph:
  - supervised operator learning task
  - compare a simple CNN, a U-Net, and an FNO
  - goal is not only low test loss but understanding why architectures differ

### 13. Problem Setup and Data
- State the Darcy PDE and boundary condition.
- Define the operator-learning objective `a -> u`.
- Briefly describe:
  - the provided training and test datasets
  - internal validation split during development
  - retraining on the full training set for final runs

### 14. Model Architectures
- Separate subsections for:
  - simple CNN
  - U-Net
  - FNO
- For each model, explain:
  - the architecture
  - why that architecture was chosen
  - optimizer and learning-rate choice
- For the CNN comparison:
  - explain why the simple CNN is a strong baseline
  - explain why U-Net skip connections should help preserve multiscale spatial structure
- For the FNO:
  - explain spectral convolutions conceptually
  - explain why FNO should be well suited to operator learning on grids

### 15. Hyperparameter Tuning Strategy
- Include a real tuning section here as well.
- Explain the manual grid search approach:
  - validation-based selection
  - final retraining on the full training set
- State the search spaces explicitly:
  - **Simple CNN / U-Net**
    - `base_channels`: `16, 32`
    - `learning_rate`: `1e-3, 5e-4`
    - `weight_decay`: `0, 1e-6`
  - **FNO**
    - `modes`: `8, 12`
    - `width`: `24, 32`
    - `learning_rate`: `1e-3, 5e-4`
- Explain the chosen final settings:
  - Simple CNN: `base_channels=32`, `lr=1e-3`, `weight_decay=1e-6`
  - U-Net: `base_channels=32`, `lr=1e-3`, `weight_decay=1e-6`
  - FNO: `modes=8`, `width=32`, `lr=5e-4`, `weight_decay=1e-6`
- Mention that, unlike Problem 1, this tuning was manual rather than Optuna-driven.

### 16. Final Quantitative Comparison
- Include a clean comparison table with:
  - model
  - epochs
  - device
  - parameter count
  - training time
  - final test loss
- Canonical values to foreground:
  - Simple CNN: about `0.00503`
  - U-Net: about `0.00299`
  - FNO on CPU: about `0.000772`
- Mention the alternative FNO MPS run:
  - about `0.0391`
- State clearly:
  - U-Net outperformed the simple CNN
  - CPU FNO was best overall

### 17. Graphical Analysis of the Darcy Predictions
- For each model, include and interpret:
  - train/test loss curve
  - truth contour
  - prediction contour
  - absolute error contour
- The analysis should be explicit:
  - simple CNN:
    - what broad structures it captures well
    - where it smooths or misses local detail
  - U-Net:
    - how skip connections improve fidelity
    - whether the error becomes more localized
  - FNO:
    - how closely the prediction follows the truth
    - whether the absolute error field is lower and more uniform
- Tie the visual analysis back to the quantitative table.

### 18. Backend Sensitivity and Numerical Reliability of the FNO
- This should be a dedicated subsection.
- Explain:
  - the FNO produced FFT warnings on MPS
  - CPU rerunning with the same hyperparameters was far better
- Use this as a serious analytical result, not a side remark.
- Explicitly compare:
  - MPS FNO final test loss `~0.0391`
  - CPU FNO final test loss `~0.000772`
- Explain the conclusion:
  - the FNO method itself is strong
  - the MPS FFT backend was the likely issue
- This section strengthens the report because it shows you validated the result instead of blindly reporting the first run.

### 19. Problem 2 Discussion
- Compare the three models conceptually:
  - simple CNN as a robust baseline
  - U-Net as a better multiscale CNN
  - FNO as the strongest operator-learning model when run on a reliable backend
- Explain why the ordering makes sense:
  - skip connections help U-Net outperform the plain CNN
  - FNO captures global structure efficiently
- Also note that fixed-grid image-to-image tasks can already be very well handled by CNNs, so the strong CNN performance is not surprising.

### 20. Problem 2 Improvements
- Add a reflective subsection based on [problem_2_improvements.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/problem_2_improvements.md).
- Include the most useful future improvements:
  - wider and deeper hyperparameter searches
  - more systematic epoch tuning
  - stronger validation/model-selection protocol
  - more extensive FNO tuning on CPU
  - architecture ablations
  - more qualitative examples and broader error summaries
- Mention that backend reliability checks should have been performed earlier for the FNO.

### 21. Problem 2 Conclusion
- One short concluding paragraph:
  - simple CNN was already strong
  - U-Net improved further
  - CPU FNO performed best overall
  - backend validation was essential for a trustworthy final comparison

---

## 22. Short Overall Conclusion
- One closing section for the whole report.
- Summarise the major takeaways:
  - Problem 1: physics-only residual minimization was not enough; small measurement data was transformative
  - Problem 2: model architecture mattered strongly; FNO was best once numerical backend issues were handled
- End with one sentence on the broader lesson:
  - numerical method choice, optimization strategy, and hardware/backend all materially affect scientific conclusions in data-driven PDE solving.

## Figures and Tables to Include
- **Include generously**, as long as each figure is analysed.
- **Problem 1 main text**
  - training loss curves for no-data and data-assisted PINNs
  - loss-component plots
  - `sigma_11` plots
  - FEM-vs-PINN displacement plots
  - FEM-vs-PINN stress plots
  - residual distributions and/or residual spatial maps
  - Optuna optimization history
  - Optuna hyperparameter importance
  - Optuna parallel coordinate
  - float32-vs-float64 rerun comparison plot
  - one summary table of final no-data vs data-assisted metrics
- **Problem 2 main text**
  - train/test loss curves for all three models
  - truth/prediction/error contour plots for all three models
  - one final quantitative comparison table
  - one explicit CPU-vs-MPS FNO comparison table or small figure
- **Appendix**
  - extra Optuna plots such as contour, timeline, EDF, intermediate values
  - larger tuning tables
  - secondary runs or supplementary figures

## Writing Rules
- Do not write a long abstract or formal front matter.
- Keep the prose technical and explanatory.
- Every important plot must answer:
  - what is being shown
  - what pattern matters
  - what it implies about the method
- Every important table must be interpreted in prose.
- Use equations where they directly support grading:
  - Problem 1 PDEs, BCs, and loss
  - Problem 2 operator-learning formulation
- Use the improvement sections to show critical reflection and awareness of limitations.

## Assumptions and Defaults
- The report is **long-form and detailed**, not tightly page-limited.
- Main structure is **by problem**, with clear subheadings aligned to coursework parts.
- Weak or messy results should be **included and analysed**, not hidden.
- Canonical simple CNN result is the `300`-epoch MPS run.
- Canonical FNO result is the `500`-epoch CPU run; the MPS run is included to explain backend sensitivity.
- If a figure is relevant and can be analysed meaningfully, include it rather than trimming aggressively.
