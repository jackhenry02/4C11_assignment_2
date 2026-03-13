# Problem 1 Hyperparameters

This note collects the main tunable settings for the Problem 1 PINN in [PINN_v2.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_2/PINN_v2.ipynb), together with a short explanation of what each one does.

## Current baseline settings

These are the main values currently used in the notebook.

- `LEARNING_RATE = 1e-3`
- `LR_STEP_SIZE = 10000`
- `LR_GAMMA = 0.5`
- `DISP_LAYERS = [2, 300, 300, 2]`
- `STRESS_LAYERS = [2, 400, 400, 3]`
- `ACTIVATION = nn.Tanh`
- `LOSS_FUNCTION = nn.MSELoss()`
- `FULL_EPOCHS = 50000`
- `TEST_EPOCHS = 1000`
- `DATA_LOSS_WEIGHT = 100.0`
- `MEASUREMENT_POINT_COUNT = 50`
- Optimizer: `Adam`
- Scheduler: `StepLR`

## Core training hyperparameters

### `LEARNING_RATE`

Controls the step size used by the optimizer.

- Too large: training can oscillate or diverge.
- Too small: training becomes slow and may stall.
- Good first values to try: `1e-3`, `5e-4`, `1e-4`.

### `FULL_EPOCHS`

Sets the main training budget for the final run.

- Larger values usually help PINNs, but increase runtime.
- The coursework brief suggests PINNs may need `50k-100k` epochs.

### `TEST_EPOCHS`

Sets the short smoke-test budget.

- This is mainly for debugging and checking that the notebook runs end-to-end.
- It is not a physics hyperparameter, but it is useful for quick experiments.

### `LOSS_FUNCTION`

Currently this is `nn.MSELoss()`.

- This is appropriate because the notebook is minimizing residuals and boundary mismatches.
- It is sensible to keep this fixed unless there is a strong reason to change it.

## Scheduler hyperparameters

### `LR_STEP_SIZE`

How many epochs pass before the learning rate is decayed.

- Larger value: learning rate stays high for longer.
- Smaller value: learning rate decays earlier.

### `LR_GAMMA`

The multiplicative decay factor used by `StepLR`.

- `0.5` means the learning rate is halved at each step.
- Smaller values decay more aggressively.

### Scheduler on/off

This is also a tuning choice, even though it is not currently exposed as a separate variable.

- With a scheduler: usually more stable late-stage optimization.
- Without a scheduler: simpler baseline and sometimes adequate.

## Network architecture hyperparameters

### `DISP_LAYERS`

Architecture of the displacement network.

- Example: `[2, 300, 300, 2]` means 2 inputs, two hidden layers of width 300, and 2 outputs.
- Increasing width or depth gives more capacity, but also higher cost and sometimes harder optimization.

### `STRESS_LAYERS`

Architecture of the stress network.

- Example: `[2, 400, 400, 3]` means 2 inputs, two hidden layers of width 400, and 3 stress outputs.
- This network often benefits from slightly higher capacity than the displacement network.

### `ACTIVATION`

Currently `nn.Tanh`.

- `tanh` is a standard PINN choice because it is smooth and differentiable.
- It is usually a better fit here than piecewise-linear activations such as ReLU.

## Data-assisted training hyperparameters

These matter mainly for part (e).

### `USE_DATA_PART_E`

Turns the extra displacement-data loss on or off.

- `False`: pure physics-informed training.
- `True`: physics-informed training plus 50 measured displacement points.

### `DATA_LOSS_WEIGHT`

Scales the contribution of the extra displacement-data term.

- Too small: the model may ignore the measurements.
- Too large: the model may fit the measurements at the expense of the PDE and boundary conditions.
- Good coarse values to test: `10`, `100`, `1000`.

### `MEASUREMENT_POINT_COUNT`

How many FEM displacement samples are used for part (e).

- The coursework asks for 50 points.
- If experimenting outside the strict coursework setup, this can be varied to study sensitivity.

## Loss-weighting hyperparameters

These are not implemented yet as separate variables, but they are worth considering if the loss components are badly imbalanced.

### `w_eq`

Weight on the equilibrium residual loss.

### `w_cons`

Weight on the constitutive-consistency loss in the interior.

### `w_cons_bc`

Weight on the constitutive-consistency loss at the boundary.

### `w_bc`

Weight on the total boundary-condition loss.

### `w_data`

Weight on the part (e) displacement-data loss.

A weighted total loss would look like

```python
loss = (
    w_eq * loss_eq
    + w_cons * loss_cons
    + w_cons_bc * loss_cons_bc
    + w_bc * loss_bc
    + w_data * loss_data
)
```

This is often a good tuning direction for PINNs because different terms naturally live on different scales.

## Optimizer hyperparameters

### Optimizer choice

Currently the notebook uses `Adam`.

- `Adam` is a strong baseline and is simple to implement.
- `LBFGS` can be useful later for PINN fine-tuning, but it is more expensive and requires a closure.

### Two-phase optimization: `Adam -> LBFGS`

This is not implemented yet, but it is a reasonable future improvement.

Useful tunable parameters would be:

- `ADAM_PHASE_EPOCHS` or `LBFGS_START_EPOCH`
- `LBFGS_LR`
- `LBFGS_MAX_ITER`
- `LBFGS_HISTORY_SIZE`

The most important of these is the switch point.

- Early switch: more refinement, less exploratory Adam training.
- Late switch: more robust early training, less time for LBFGS refinement.

## Numerical and runtime settings

These are not pure model hyperparameters, but they strongly affect behavior.

### `FLOAT_BITS`

Controls whether the model uses `float32` or `float64`.

- `float64` is usually safer for derivative-heavy PINNs.
- `float32` can be much faster on MPS, and is useful for quick tests.

### `FORCE_DEVICE`

Controls whether the selected accelerator should still be used if `float64` is unsupported.

- `True`: keep the accelerator and drop to `float32`.
- `False`: fall back to CPU `float64`.

### `RANDOM_SEED`

Improves reproducibility.

- Useful for fair hyperparameter comparisons.
- Not a performance hyperparameter in itself.

## Experiment-management settings

These do not change the learned solution much, but they matter for practical workflow.

### `CHECKPOINT_EVERY`

How often the notebook saves a resume checkpoint.

- Smaller value: safer, more disk writes.
- Larger value: less overhead, higher risk of losing progress.

### `PRINT_EVERY`

How often the training loop prints progress.

- Useful for monitoring.
- Too frequent printing can add a bit of overhead on long runs.

## What is not a hyperparameter here

These are physical problem parameters from the coursework setup and should normally be kept fixed.

- `YOUNGS_MODULUS = 10.0`
- `POISSON_RATIO = 0.3`
- `RIGHT_TRACTION = 0.1`
- `TOP_TRACTION = 0.0`

Changing these changes the PDE being solved, not just the training procedure.

## Suggested tuning priority

If tuning manually, a sensible order is:

1. `LEARNING_RATE`
2. `DISP_LAYERS` and `STRESS_LAYERS`
3. `FULL_EPOCHS`
4. Scheduler settings: `LR_STEP_SIZE`, `LR_GAMMA`, or scheduler on/off
5. `DATA_LOSS_WEIGHT` for part (e)
6. Loss weights such as `w_bc` and `w_data`
7. Two-phase optimizer switch point if `Adam -> LBFGS` is added later

## Practical recommendation

For quick iteration:

- use `float32` on MPS
- use `TEST_RUN = True`
- keep the baseline optimizer as Adam

For final coursework-quality runs:

- use `float64`
- run the full epoch budget
- compare a baseline run and a part (e) run
- only then consider more advanced tuning such as loss weighting or `Adam -> LBFGS`
