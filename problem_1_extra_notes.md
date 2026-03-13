# Problem 1 notes


## To fill in:

I read [PINN.ipynb](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs and Courswork/IIB Coursework/4C11/4C11_assignment_2/PINN.ipynb). For Problem 1, the skeleton is mostly complete already. The main fill-ins are in the setup cell and the training-loop cell.

What is fixed by the coursework brief or the elasticity equations:
- `E = 10` and `mu = 0.3`
- `dvdx` is the gradient of `v` with respect to `x`
- `e_12` is the engineering shear strain term `0.5 * (du/dy + dv/dx)`
- `v_bc`, `dvdx_bc`, and `e_12_bc` are the boundary versions of the same quantities
- `sig_22 = sigma[:,1]` and `sig_12 = sigma[:,2]`
- `dsig22dx` and `dsig12dx` are gradients of those stress components
- `eq_x2` is the second equilibrium equation `d sigma_12 / dx + d sigma_22 / dy`
- `loss_eq2` is the loss against zero body force for that second equilibrium residual
- `tau_R = 0.1` and `tau_T = 0.0`
- `sig_T = stress_net(T_boundary)` and `sig_C = stress_net(C_boundary)`

What is a modelling/training choice rather than a fixed value:
- `iterations`
- `loss_func`
- `optimizer`
- `scheduler`

If you want to keep the notebook very close to the skeleton, the natural choices are:
- `loss_func` as MSE
- `optimizer` as Adam on both nets’ parameters
- some simple LR scheduler, or even no scheduler initially
- `iterations` in the tens of thousands, since the brief explicitly says PINNs often need `50k-100k`

There are also three small non-placeholder gaps to be aware of:
- The notebook loads `Plate_data.mat`, but the file in the repo is `plate_data.mat`. On a case-sensitive filesystem that will fail.
- It prints the loss each epoch, but it does not store a loss history, so the required training-error-vs-epoch plot is not implemented yet.
- The `device` setup is not actually wired into the tensors or networks, so despite the MPS/CUDA check, this notebook still runs on CPU unless that is added later.

For part (e), the data-assisted version is already sketched: you just uncomment the `x_fix`, `u_fix`, `loss_fix` block and include it in the total loss.

If you want, next I can go through the training cell line by line and tell you exactly what each loss term is doing before we fill anything in.


## Spikey loss function: schedular or float bits

The loss function was very spikey and settled down each time the learning rate was reduced. I was wondering whether I should try a different scheduler, or just explore the ideal hyperparameters for the stepLR scheduler, or simply tru the float64 version and see if that works

My recommendation is: try the `float64` run first, with the same scheduler and the same model settings.

Why:
- Your `float32` MPS run got a very low training loss, but the FEM agreement was still weak.
- The loss spikes settling after each LR drop suggest the optimizer was overshooting and then recovering.
- That is consistent with `float32` numerical noise and an initial LR that is a bit aggressive, not necessarily with the wrong scheduler family.

So I would not switch scheduler yet. I’d do this in order:

1. Run `float64` with the current `StepLR(step_size=10000, gamma=0.5)` unchanged.
2. If the `float64` run is still spiky or FEM agreement is still poor, keep `StepLR` and lower `LEARNING_RATE` first, probably to `5e-4`.
3. Only after that would I tune `LR_STEP_SIZE` and `LR_GAMMA`.
4. I would try a different scheduler only if the `float64` + lower-LR runs still behave badly.

Why I’d keep `StepLR` for now:
- It is simple and defensible in coursework.
- Your run already showed that each decay helped.
- That points more toward “schedule strength / initial LR” than “wrong scheduler type”.

If you want the most informative next experiment, do exactly one change:
- `float64`, everything else unchanged.

If you want the most likely better-behaved configuration after that, it would be:
- `float64`
- `LEARNING_RATE = 5e-4`
- keep `step_size = 10000`, `gamma = 0.5`

I would avoid `ReduceLROnPlateau` for now because your training loss is noisy and you do not have a clean validation objective. Cosine scheduling is possible, but it is less interpretable than a simple step decay for this report.