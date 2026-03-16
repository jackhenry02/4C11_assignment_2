# Problem 2 Improvements

This document summarises ways the Problem 2 Darcy workflow could be improved if more time, compute, and experimentation budget were available. It is intentionally broader than "what was needed to make the coursework work". It includes improvements to data splitting, model design, optimization, tuning, hardware choices, validation, and reporting.

## 1. Improve The Hyperparameter Search Budget

- Run larger hyperparameter studies instead of small manual grids only.
- Expand the search over:
  - learning rate
  - scheduler on/off
  - scheduler step size and decay
  - batch size
  - epoch count
  - model width and depth
- Use a second-stage search around the best region rather than one coarse grid only.
- Tune for the same training budget used in the final runs, rather than tuning on shorter proxy runs and extrapolating.
- Repeat promising configurations across multiple random seeds.
- Compare the best few settings, not just the single best setting from one search.

## 2. Tune Epoch Count More Systematically

- Treat training duration as a real hyperparameter rather than fixing:
  - simple CNN at `300`
  - U-Net at `400`
  - FNO at `500`
- Run small continuation experiments to determine whether each model has actually plateaued.
- Compare validation and test performance at several budgets, for example:
  - `200`
  - `300`
  - `400`
  - `500`
  - `700`
- Use the same epoch budget across models if strict training-budget fairness is desired.
- Alternatively, justify model-specific epoch counts explicitly as tuned training hyperparameters.

## 3. Improve The Validation Protocol

- Keep the current `85/15` validation split for tuning, but repeat it with multiple random splits to reduce dependence on one partition.
- Use cross-validation on the training set if more rigorous model selection is required.
- Record both:
  - best validation loss
  - final validation loss
- Compare selecting the best checkpoint against selecting the final checkpoint.
- Use the same fixed validation split for all model comparisons to reduce variance.
- Save and report the exact split indices for reproducibility.

## 4. Improve Final Model Selection

- Use the validation-best checkpoint more explicitly in the final comparison, not only the final full-train run.
- Compare:
  - validation-best model
  - final retrained model on full training data
- For the final report, explain the distinction between:
  - tuning on validation
  - retraining on all training data
  - evaluating once on the fixed test set
- Store a small comparison table for each model showing how much retraining on the full training set changed performance.

## 5. Improve The Classical CNN Baseline

- Explore an even wider range of classical CNN baselines:
  - shallow same-resolution CNN
  - encoder-decoder without skips
  - residual CNN
- Tune the number of encoder-decoder levels.
- Compare:
  - `base_channels = 16`
  - `32`
  - `64`
- Compare batch normalization against no normalization.
- Compare bilinear upsampling against transposed convolution.
- Compare `GELU` against `ReLU` and `SiLU`.
- Test whether the classical CNN benefits from mild residual connections.

## 6. Improve The U-Net Architecture

- Tune the number of levels in the U-Net.
- Increase base channel width if memory allows.
- Compare concatenation skips against additive skips.
- Add residual blocks inside encoder and decoder stages.
- Test deeper bottlenecks or a larger bottleneck width.
- Compare bilinear upsampling against learnable transposed convolutions.
- Explore light regularization such as:
  - small weight decay
  - dropout in the bottleneck only
- Compare whether batch normalization is optimal, or whether group normalization works better.

## 7. Improve The FNO Architecture

- Tune `modes` more broadly, for example:
  - `6`
  - `8`
  - `10`
  - `12`
  - `16`
- Tune `width` more broadly, for example:
  - `16`
  - `24`
  - `32`
  - `48`
- Tune the number of Fourier blocks.
- Tune the width of the pointwise MLP sublayers.
- Compare different activation placements in the Fourier blocks.
- Compare the current skeleton-led FNO against a slightly more expressive but still custom implementation.
- Explore spectral truncation strategies to see whether the current number of retained modes is too restrictive or too noisy.

## 8. Investigate The FNO Hardware Backend More Carefully

- The MPS backend produced FFT-related warnings for the FNO but not for the CNNs.
- A CPU FNO run performed substantially better than the MPS FNO run with the same or similar settings.
- This suggests the FNO should be hardware-validated separately from the CNNs.
- Improvements here would include:
  - compare MPS and CPU directly for the same FNO configuration
  - rerun the top few FNO configurations on CPU
  - base the final FNO result on the more reliable backend
  - document the backend choice in the report
- If time allowed, use CPU-only tuning for the FNO or a backend without FFT warnings.

## 9. Improve The Optimizer Strategy

- Tune `Adam` learning rate more systematically.
- Tune batch size jointly with learning rate.
- Compare:
  - fixed learning rate
  - `StepLR`
  - cosine decay
  - `ReduceLROnPlateau`
- Step schedulers based on validation improvement rather than only epoch count.
- Add warm-up for the FNO if optimization is unstable early on.
- Compare `Adam` against `AdamW`.
- Compare small weight decay values more systematically.

## 10. Add More Robust Experiment Tracking

- The current per-run tracking is already useful, but it could be extended further.
- Save:
  - exact package versions
  - PyTorch backend info
  - device name
  - random seed
  - dataset split indices
- Save a compact CSV or JSON summary table across all runs for quick model comparison.
- Add a small results notebook or script that aggregates all tracked runs automatically.
- Record whether each result came from:
  - development mode
  - grid search
  - final full-train mode
  - CPU FNO check

## 11. Improve Reproducibility

- Fix all random seeds consistently for:
  - NumPy
  - PyTorch
  - split generation
  - data loader shuffling
- Save the exact train/validation index split for each development run.
- Save the exact selected hyperparameters in versioned JSON files.
- Record the exact notebook filename used, especially since there are now variants such as:
  - `DARCY_FNO_v2.ipynb`
  - CPU check notebooks
- Export a minimal run manifest that can reproduce the final report runs exactly.

## 12. Improve Data Handling And Preprocessing

- Compare the current Gaussian normalization of `a` against alternative input scalings.
- Consider normalizing `u` during optimization rather than only decoding before loss evaluation.
- Compare training on normalized outputs against raw-output loss.
- Check whether spatial coordinate augmentation is equally useful for the CNNs as for the FNO.
- Explore whether the CNNs benefit from being given coordinate channels explicitly.
- Investigate whether any samples in the test set are systematically harder than others.

## 13. Improve The Evaluation Metrics

- Report more than one scalar metric.
- In addition to relative `L2`, report:
  - RMSE
  - max absolute error
  - mean absolute error
  - per-sample error distribution
- Report summary statistics across the whole test set, not only one sample.
- Add quantiles of test error, for example:
  - median
  - 90th percentile
  - worst-case sample
- Add a table of final train/validation/test losses for each model.

## 14. Improve The Visual Evaluation

- Plot more than one test sample in the final report.
- Include:
  - best-case sample
  - median sample
  - difficult sample
- Add error contour plots in the main model comparison section, not only truth and prediction.
- Use consistent contour levels across models for the same sample.
- Add a side-by-side comparison figure across all three models on the same test sample.
- Include the coefficient field `a(x, y)` alongside the solution `u(x, y)` for the chosen sample so the reader can relate input difficulty to output quality.

## 15. Improve Fairness Of Model Comparison

- Make the comparison protocol more explicit:
  - same train/test split
  - same validation procedure
  - same tuning philosophy
- Decide whether fairness means:
  - same epoch count
  - same compute budget
  - same tuning effort
- If strict fairness is required, retrain all models at the same epoch count.
- If model-specific epochs are kept, justify them as tuned training hyperparameters rather than arbitrary choices.
- Consider comparing wall-clock time as well as test error.
- Compare parameter counts and possibly error per parameter.

## 16. Improve The Classical-CNN vs U-Net Story

- The current setup already uses a stronger classical CNN than the original skeleton implied.
- If more time were available, an ablation could isolate what made the U-Net better:
  - skip connections only
  - depth only
  - channel width only
  - normalization only
- This would strengthen the report by showing whether the gain comes mainly from:
  - the encoder-decoder structure
  - the skip connections
  - greater capacity

## 17. Improve The FNO-vs-CNN Interpretation

- Since the FNO underperformed the CNN/U-Net in this fixed-grid setting, the report could be strengthened by additional analysis:
  - whether the FNO was under-tuned
  - whether backend choice harmed it
  - whether fixed-resolution image-to-image mapping simply favors CNNs here
- A stronger FNO evaluation would include:
  - CPU vs MPS comparison
  - a slightly broader CPU hyperparameter sweep
  - a short discussion of why FNOs can be more attractive when resolution transfer matters, even if they are not best here

## 18. Improve The Search Methodology

- Manual grid search was sensible for coursework, but it can still be improved.
- Use:
  - a coarse grid first
  - then a local refinement around the best settings
- Reduce the number of obviously weak combinations once patterns emerge.
- For the FNO specifically, use a CPU-only refinement grid on the top few MPS candidates.
- If time allowed, use Optuna or another search tool for:
  - U-Net
  - FNO
- If Optuna is used later, keep the objective aligned with the report metric and preserve the internal validation protocol.

## 19. Improve The Notebooks As Research Artifacts

- Add a cell that prints the selected hyperparameters in a clean summary block at the end.
- Add a cell that loads and displays the metrics from the saved run folder.
- Add a comparison notebook that reads all three final runs and produces a single summary table and figure set.
- Add a quick “sanity check” section that plots a few raw input coefficient fields before training.
- Add a cell that prints the best validation epoch and the final epoch.

## 20. Improve Reporting Readiness

- Create one clean comparison table with:
  - model name
  - parameter count
  - training time
  - final test relative `L2`
- Save report-ready figures with consistent naming and dimensions.
- Use consistent sample indices across all model plots.
- Add a short written justification for:
  - why the U-Net was chosen as the main CNN
  - why the FNO backend changed to CPU if that is used in the final result
- Make the difference between development runs and final full-train runs explicit in the report.

## 21. Highest-Value Improvements If Time Were Limited

If only a few improvements could be made, these would likely give the best return:

- Run a small CPU-only refinement sweep for the FNO around the best MPS-found configuration.
- Add a clean final comparison table across the three models.
- Report multiple test-sample contour plots instead of just one.
- Tune epoch count more systematically, or justify the current model-specific values more explicitly.
- Compare best-validation checkpoints against final full-train runs for each model.
- Add CPU/MPS backend commentary for the FNO to explain the hardware sensitivity.

## 22. Bottom Line

The current Problem 2 workflow is already solid enough for coursework:

- a classical CNN baseline exists
- a stronger U-Net CNN exists
- an FNO exists
- internal validation was used for tuning
- final runs were retrained and saved with figures and metrics

The main limitations are:

- epoch count was not tuned as systematically as other hyperparameters
- the FNO appears hardware-sensitive, especially on MPS
- the FNO search was still relatively small
- final comparisons could be made more rigorous with broader metrics and more than one sample

If more time were available, the single biggest upgrade would likely be a CPU-based refinement of the FNO plus a more systematic final comparison protocol across all three models.
