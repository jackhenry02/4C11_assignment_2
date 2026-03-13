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