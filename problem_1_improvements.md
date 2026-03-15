# Problem 1 Improvements

This document summarises ways the Problem 1 PINN workflow could be improved if more time and compute were available. It is deliberately broader than "what was necessary to make the coursework work". It includes improvements to the mathematical formulation, hyperparameter tuning, numerical stability, validation, and reporting.

## 1. Improve The Hyperparameter Search Budget

- Run the Optuna studies with a longer proxy budget than `10000` epochs.
- Use a second-stage hyperparameter search around the best region rather than one broad first pass.
- Run the final hyperparameter comparisons at the full training budget, for example `50000` or `100000` epochs, instead of extrapolating from shorter runs.
- Tune with the same precision and device intended for the final results, rather than using `float32` as a screening proxy and `float64` later.
- Repeat top trials with multiple random seeds to avoid selecting a configuration that only happened to work well once.
- Disable pruning for a subset of promising trials to check whether apparently weak early trials become strong late trials.
- Use a larger number of completed trials so that more of the search space is explored before drawing conclusions.

## 2. Use A Better Tuning Objective

- Optimize a composite objective instead of displacement error alone.
- Combine displacement and stress metrics in a single score, for example:
  - displacement relative `L2`
  - `sigma11` relative `L2`
  - full stress relative `L2`
- Weight the composite objective according to the coursework priorities, especially because the final required visualisation is a stress field.
- Use a Pareto or multi-objective study instead of forcing everything into one scalar objective.
- Evaluate the objective using the best checkpoint within a run, not only the final checkpoint.
- Include boundary-condition residual quality in the validation objective, not just FEM agreement.
- Include equilibrium and constitutive residual summaries in the selection metric so a model cannot win purely by matching displacement while violating the PDE more strongly.

## 3. Improve Loss Design

- Tune all major loss weights systematically, not only `w_bc`, `w_cons_bc`, and `w_data`.
- Add a separate weight for equilibrium and constitutive losses if their raw scales differ strongly.
- Separate the boundary-condition weight into:
  - symmetry BC weight
  - traction BC weight
  - hole traction-free BC weight
- Reweight the hole boundary specifically because stress concentration near the hole is the hardest part of the problem.
- Use adaptive loss weighting rather than fixed hand-tuned constants.
- Try gradient-balanced or NTK-style loss balancing methods so one loss term does not dominate optimization.
- Normalize each loss term by a meaningful physical scale or by running statistics.
- Penalize stress mismatch more strongly if the report emphasizes stress-field accuracy over displacement accuracy.

## 4. Improve The Optimizer Strategy

- Add a two-stage optimizer:
  - `Adam` for coarse training
  - `LBFGS` for final refinement
- Tune the Adam-to-LBFGS switch point as a hyperparameter.
- Tune the LBFGS settings:
  - learning rate
  - history size
  - max iterations per step
- Compare no scheduler against late-stage scheduling at the full budget.
- If using Adam only, test smaller learning rates for longer runs.
- Use a long-run schedule tailored to `100000` epochs rather than inheriting a short-run schedule.
- Consider manual learning-rate drops triggered by the FEM validation metrics rather than only by epoch count.

## 5. Improve The Network Architecture

- Tune depth as well as width.
- Compare:
  - two hidden layers
  - three hidden layers
  - wider but shallower networks
  - narrower but deeper networks
- Try a single network outputting both displacement and stress instead of two separate networks.
- Compare separate displacement and stress networks against a shared trunk with two output heads.
- Test other activations suitable for PINNs, such as:
  - `tanh`
  - `sine` / SIREN-style networks
  - softplus
- Tune weight initialization explicitly rather than using PyTorch defaults only.
- Consider residual or skip connections if deeper networks are explored.

## 6. Improve The Treatment Of The Data Points In Part (e)

- Use a train/validation split for the 50 measurements.
- Tune `w_data` against held-out measurement points rather than using all 50 only for training.
- Compare several measurement counts, for example `10`, `25`, `50`, and `100`.
- Choose measurement points more intelligently rather than uniformly random sampling.
- Bias measurements toward difficult regions, especially near the hole boundary where stress varies rapidly.
- Compare random measurement selection against stratified spatial sampling.
- Average over multiple measurement subsets to reduce sensitivity to one lucky or unlucky sample.

## 7. Improve Sampling Of Collocation And Boundary Points

- Resample interior collocation points during training instead of keeping one fixed set.
- Oversample regions near the hole where stress gradients are largest.
- Use adaptive sampling driven by residual magnitude.
- Increase the density of boundary points on the circular hole boundary.
- Separate interior and boundary mini-batching if full-batch training becomes too slow.
- Use curriculum-style sampling:
  - coarse domain coverage first
  - more concentrated sampling near difficult regions later

## 8. Improve Numerical Stability And Precision

- Run all final selection and final report results in `float64`.
- Benchmark `float64` CPU against `float32` MPS more systematically across several trials, not only a few reruns.
- Use a stricter protocol for deciding when `float32` is acceptable.
- Profile whether the poor `float32` behavior comes from MPS itself, from the learning rate, or from derivative noise.
- Consider CPU-only validation runs for the exact final figures if accelerator precision is questionable.
- Add NaN and Inf detection with more informative per-loss diagnostics.
- Save gradient norms during training to diagnose instability.

## 9. Improve Data And Input Scaling

- Normalize the spatial coordinates before they enter the network, for example to `[0, 1]` or `[-1, 1]`.
- Compare normalized vs unnormalized coordinates explicitly.
- Normalize output magnitudes or stress scales where appropriate.
- Scale the traction values and material constants consistently if a dimensionless formulation is adopted.
- Reformulate the problem in non-dimensional form so losses are more naturally balanced.

## 10. Improve The Physical Formulation

- Explore hard enforcement of simple boundary conditions such as symmetry constraints, rather than enforcing them only in the loss.
- Encode some BCs into the trial solution ansatz so the optimizer has fewer tasks to learn.
- Investigate whether the separate stress network is strictly necessary, or whether stress reconstructed from displacement alone performs better.
- Compare plane stress against plane strain only as a sensitivity check, even though plane stress is the correct coursework model.
- Compare the PINN solution to the analytical Kirsch solution for a plate with a circular hole, at least qualitatively along selected lines.
- Use analytical stress concentration factors as an additional sanity check.

## 11. Improve Validation Against Reference Solutions

- Use more than one evaluation metric.
- Report:
  - displacement relative `L2`
  - stress relative `L2`
  - RMSE
  - max absolute error
  - boundary residual statistics
  - equilibrium residual statistics
  - constitutive residual statistics
- Compare models using the best checkpoint rather than the final checkpoint only.
- Compare baseline and part (e) at the same precision and epoch budget.
- Compare results across multiple seeds.
- Plot error maps for stress as well as displacement.
- Add line plots through critical regions, for example:
  - along the symmetry axes
  - around the hole boundary
  - along the right boundary under traction

## 12. Improve Stress Evaluation

- Use stress metrics more prominently during tuning, because stress is one of the most important outputs of the problem.
- Compare different stress-selection criteria:
  - best displacement model
  - best stress model
  - best composite model
- Validate stress specifically near the hole edge where stress concentration matters most.
- Report the peak `sigma11` around the hole and compare it to FEM and analytical expectations.
- Add elementwise stress error histograms.
- Add spatial stress-error contour maps for all three stress components, not only summaries.

## 13. Improve Pruning Strategy

- Use a more conservative pruner so fewer late-blooming trials are discarded.
- Increase `n_startup_trials` so the pruning baseline is based on more complete runs.
- Increase `n_warmup_steps` so pruning starts later than `500` epochs.
- Report more checkpoints during training if finer-grained pruning decisions are desired.
- Alternatively, report less frequently if early metrics are too noisy.
- Compare the median pruner against no pruning for a subset of trials.
- Manually rerun promising pruned trials that had strong intermediate values.

## 14. Improve The Final Notebook Runs

- Add an explicit configuration block saying which tuned trial is being reproduced and why.
- Record the exact Optuna trial ID and rerun comparison metrics inside the notebook output.
- Add an option to load a tuned config from a JSON file so the notebook and tuning script cannot drift apart.
- Save the best checkpoint and the final checkpoint separately, then compare them automatically.
- Resume from checkpoints automatically if a run is interrupted.
- Save intermediate figures during long runs, not only at the end.
- Store training summaries in a single comparison table for baseline and part (e).

## 15. Improve Reproducibility

- Lock all random number generators consistently for:
  - PyTorch
  - NumPy
  - measurement-point sampling
  - Optuna trials
- Record exact package versions alongside each final run.
- Save the full tuned config and evaluation metrics in a small results manifest.
- Avoid reusing a shared Optuna database across experimental phases unless the study naming is strictly controlled.
- Export top trial configs into versioned JSON files for the final notebook runs.

## 16. Improve Runtime And Efficiency

- Profile the training loop to identify the most expensive autograd operations.
- Reduce repeated forward passes on boundaries where possible.
- Explore interior-point mini-batching while keeping boundary terms full-batch.
- Test whether `torch.compile` helps in this environment.
- Use a script-based training path for long runs rather than notebooks only, to reduce UI overhead and improve reproducibility.
- Schedule long runs overnight with checkpointing and automatic resume.

## 17. Improve The Final Report Narrative

- Explain clearly that PINN training loss and FEM agreement are not the same thing.
- Distinguish between:
  - the internal training objective
  - the external validation metrics
- Explain why displacement-only tuning was used first and what its limitations are.
- Justify the final trial choice, especially for part (e), where displacement-optimal and stress-optimal configurations can differ.
- Include an ablation table showing what mattered most:
  - precision choice
  - data usage
  - boundary-loss weighting
  - scheduler on/off
  - architecture width
- Discuss why the data-assisted run improves displacement and whether it also improves stress uniformly.

## 18. Highest-Value Improvements If Time Were Limited

If only a few additional improvements could be made, these would likely give the best return:

- Run the hyperparameter tuning with a longer proxy budget, at least `15000-25000` epochs.
- Use a composite tuning objective including both displacement and stress error.
- Tune and validate entirely in `float64`.
- Compare `Adam` against `Adam -> LBFGS` for final refinement.
- Use adaptive or more systematic loss weighting, especially near the hole boundary.
- Run final comparisons over multiple random seeds or measurement subsets.

## 19. Bottom Line

The current workflow is defensible and sufficient for coursework, but it is still a practical engineering compromise. The biggest limitations are:

- hyperparameter tuning was done on short proxy runs rather than the full budget
- the tuning objective emphasized displacement more than stress
- pruning may have removed some trials that would have improved later
- final model selection still depends on tradeoffs between internal PINN losses, FEM displacement, and FEM stress

If more time were available, the most important upgrade would be to make model selection more faithful to the actual coursework goal: accurate physics and accurate stress, not just low displacement error on a short proxy run.
