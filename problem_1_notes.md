# Problem 1 notes

## (a) Governing PDE and boundary conditions

We solve a 2D linear elasticity problem for a quarter plate with a circular hole under the **plane stress** assumption.

Let the displacement field be

$$
\mathbf{u}(x,y) =
\begin{bmatrix}
u(x,y) \\
v(x,y)
\end{bmatrix},
$$

and let the Cauchy stress tensor be

$$
\boldsymbol{\sigma} =
\begin{bmatrix}
\sigma_{11} & \sigma_{12} \\
\sigma_{12} & \sigma_{22}
\end{bmatrix}.
$$

### Strong form

With zero body force, the equilibrium equations in the domain $\Omega$ are

$$
\nabla \cdot \boldsymbol{\sigma} = \mathbf{0},
$$

or componentwise

$$
\frac{\partial \sigma_{11}}{\partial x} + \frac{\partial \sigma_{12}}{\partial y} = 0,
$$

$$
\frac{\partial \sigma_{12}}{\partial x} + \frac{\partial \sigma_{22}}{\partial y} = 0.
$$

The small-strain tensor is

$$
\boldsymbol{\varepsilon} =
\begin{bmatrix}
\varepsilon_{11} \\
\varepsilon_{22} \\
\varepsilon_{12}
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\partial u}{\partial x} \\
\dfrac{\partial v}{\partial y} \\
\dfrac{1}{2}\left(\dfrac{\partial u}{\partial y} + \dfrac{\partial v}{\partial x}\right)
\end{bmatrix}.
$$

Under plane stress, the constitutive law is

$$
\begin{bmatrix}
\sigma_{11} \\
\sigma_{22} \\
\sigma_{12}
\end{bmatrix}
=
\mathbf{C}
\begin{bmatrix}
\varepsilon_{11} \\
\varepsilon_{22} \\
\varepsilon_{12}
\end{bmatrix},
$$

with

$$
\mathbf{C}
=
\frac{E}{1-\nu^2}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \dfrac{1-\nu}{2}
\end{bmatrix}.
$$

For this coursework,

$$
E = 10, \qquad \nu = 0.3.
$$

### Boundary conditions

Because only one quadrant of the plate is modelled, symmetry conditions are imposed on the left and bottom edges:

Left boundary $\Gamma_L$:

$$
u = 0.
$$

Bottom boundary $\Gamma_B$:

$$
v = 0.
$$

Uniform traction is applied on the outer right and top edges. In the coursework data,

$$
\sigma_1 = 0.1, \qquad \sigma_2 = 0.
$$

On the right boundary $\Gamma_R$, the outward normal is $\mathbf{n} = (1,0)^T$, so

$$
\boldsymbol{\sigma}\mathbf{n} =
\begin{bmatrix}
\sigma_{11} \\
\sigma_{12}
\end{bmatrix}
=
\begin{bmatrix}
\sigma_1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
0.1 \\
0
\end{bmatrix}.
$$

Hence

$$
\sigma_{11} = 0.1, \qquad \sigma_{12} = 0 \quad \text{on } \Gamma_R.
$$

On the top boundary $\Gamma_T$, the outward normal is $\mathbf{n} = (0,1)^T$, so

$$
\boldsymbol{\sigma}\mathbf{n} =
\begin{bmatrix}
\sigma_{12} \\
\sigma_{22}
\end{bmatrix}
=
\begin{bmatrix}
0 \\
\sigma_2
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}.
$$

Hence

$$
\sigma_{22} = 0, \qquad \sigma_{12} = 0 \quad \text{on } \Gamma_T.
$$

The circular hole boundary $\Gamma_C$ is traction free:

$$
\boldsymbol{\sigma}\mathbf{n} = \mathbf{0} \quad \text{on } \Gamma_C,
$$

which in components gives

$$
\sigma_{11} n_x + \sigma_{12} n_y = 0,
$$

$$
\sigma_{12} n_x + \sigma_{22} n_y = 0.
$$

So the full boundary-value problem is: find $\mathbf{u}$ such that equilibrium and the plane-stress constitutive law hold in $\Omega$, together with the symmetry, traction, and traction-free conditions above.

## (b) PINN loss function

The notebook uses two neural networks:

- `disp_net(x,y)` predicts displacement:

$$
\hat{\mathbf{u}}(x,y) =
\begin{bmatrix}
\hat{u}(x,y) \\
\hat{v}(x,y)
\end{bmatrix},
$$

- `stress_net(x,y)` predicts stress:

$$
\hat{\boldsymbol{\sigma}}(x,y) =
\begin{bmatrix}
\hat{\sigma}_{11}(x,y) \\
\hat{\sigma}_{22}(x,y) \\
\hat{\sigma}_{12}(x,y)
\end{bmatrix}.
$$

From the predicted displacement, we form the strain

$$
\hat{\boldsymbol{\varepsilon}} =
\begin{bmatrix}
\dfrac{\partial \hat{u}}{\partial x} \\
\dfrac{\partial \hat{v}}{\partial y} \\
\dfrac{1}{2}\left(\dfrac{\partial \hat{u}}{\partial y} + \dfrac{\partial \hat{v}}{\partial x}\right)
\end{bmatrix},
$$

and the corresponding constitutive stress

$$
\hat{\boldsymbol{\sigma}}^{\,a} = \mathbf{C}\hat{\boldsymbol{\varepsilon}}.
$$

This is the "augmented" stress used in the skeleton.

Let `MSE` denote the mean-squared error over the relevant collocation or boundary points. The total PINN loss in the notebook is

$$
\mathcal{L}
=
\mathcal{L}_{eq}
+ \mathcal{L}_{cons}
+ \mathcal{L}_{cons}^{bc}
+ \mathcal{L}_{bc},
$$

where:

### 1. Equilibrium loss

This enforces the PDE residuals

$$
r_1 = \frac{\partial \hat{\sigma}_{11}}{\partial x} + \frac{\partial \hat{\sigma}_{12}}{\partial y},
\qquad
r_2 = \frac{\partial \hat{\sigma}_{12}}{\partial x} + \frac{\partial \hat{\sigma}_{22}}{\partial y}.
$$

So

$$
\mathcal{L}_{eq}
=
\operatorname{MSE}(r_1,0) + \operatorname{MSE}(r_2,0).
$$

### 2. Constitutive consistency loss in the interior

This makes the stress network agree with Hooke's law applied to the displacement network:

$$
\mathcal{L}_{cons}
=
\operatorname{MSE}\left(\hat{\boldsymbol{\sigma}}^{\,a}, \hat{\boldsymbol{\sigma}}\right).
$$

### 3. Constitutive consistency loss on the boundary

The same constitutive consistency is also enforced on the union of all boundary points:

$$
\mathcal{L}_{cons}^{bc}
=
\operatorname{MSE}\left(\hat{\boldsymbol{\sigma}}^{\,a}_{bc}, \hat{\boldsymbol{\sigma}}_{bc}\right).
$$

### 4. Boundary-condition loss

This is the sum of the losses on each part of the boundary:

$$
\mathcal{L}_{bc}
=
\mathcal{L}_L + \mathcal{L}_B + \mathcal{L}_R + \mathcal{L}_T + \mathcal{L}_C.
$$

Left symmetry boundary:

$$
\mathcal{L}_L = \operatorname{MSE}(\hat{u}|_{\Gamma_L},0).
$$

Bottom symmetry boundary:

$$
\mathcal{L}_B = \operatorname{MSE}(\hat{v}|_{\Gamma_B},0).
$$

Right traction boundary:

$$
\mathcal{L}_R
=
\operatorname{MSE}(\hat{\sigma}_{11}|_{\Gamma_R},0.1)
+ \operatorname{MSE}(\hat{\sigma}_{12}|_{\Gamma_R},0).
$$

Top traction boundary:

$$
\mathcal{L}_T
=
\operatorname{MSE}(\hat{\sigma}_{22}|_{\Gamma_T},0)
+ \operatorname{MSE}(\hat{\sigma}_{12}|_{\Gamma_T},0).
$$

Traction-free circular hole:

$$
\mathcal{L}_C
=
\operatorname{MSE}(\hat{\sigma}_{11}n_x + \hat{\sigma}_{12}n_y,0)
+ \operatorname{MSE}(\hat{\sigma}_{12}n_x + \hat{\sigma}_{22}n_y,0).
$$

Putting everything together, the skeleton code uses

$$
\mathcal{L}
=
\mathcal{L}_{eq}
+ \mathcal{L}_{cons}
+ \mathcal{L}_{cons}^{bc}
+ \mathcal{L}_L + \mathcal{L}_B + \mathcal{L}_R + \mathcal{L}_T + \mathcal{L}_C.
$$

### Extension for part (e)

When displacement measurements at 50 interior points are added, a data loss is included:

$$
\mathcal{L}_{data}
=
\operatorname{MSE}\left(\hat{\mathbf{u}}(x_i), \mathbf{u}^{obs}(x_i)\right).
$$

Then the total loss becomes

$$
\mathcal{L}_{total} = \mathcal{L} + \lambda_{data}\mathcal{L}_{data},
$$

where in the notebook sketch the weight is chosen as $\lambda_{data} = 100$.
