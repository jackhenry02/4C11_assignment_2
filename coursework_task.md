# Data-driven and learning based methods

## in mechanics and materials

**Lent 2026**
Dr. Burigede Liu, University of Cambridge

---

# Project #2

**assigned:** Tuesday, Feb 24, 2026
**due:** Tuesday, March 17, 2026, 17:00

Please submit your project online through Moodle!

---

## collaboration policy

You are encouraged to discuss the homework problems and solution strategies with your classmates, but you must find the solutions and write down your answers to all problems by yourself.

---

# Problem 1: Solving PDE: Plate with a hole (45 marks)

The objective of this project is to write your own physics informed neural network (PINN) solver to solve linear elastic boundary value problems in 2D (under plane stress assumptions). Let us solve this boundary problem numerically with our PINN code. By symmetry, it is sufficient to simulate only one quadrant of the problem with uniform traction applied on two edges and sliding essential boundary conditions applied on the other two edges, as shown in Figures 1–2.

You are provided with a Matlab script **Plate_hole.m**, to generate the Collocation points to **Plate.mat** as the input to the PINN, as well as to provide you with a reference truth solution of the problem from the Finite Element method.

Young’s modulus, Poisson’s ratio and plate thickness are

* $E = 10 , \text{N/m}^2$
* $\nu = 0.3$
* $t = 1 , \text{m}$

respectively.

Use

* $\sigma_1 = 0.1 , \text{N/m}^2$
* $\sigma_2 = 0 , \text{N/m}^2$

---

**Figure 1:** Plate with circular hole

**Figure 2:** Numerical model with dimensions

---

### (a)

Write down the governing partial differential equation for this problem and specify the boundary conditions.

---

### (b)

Define the PINN loss function.

---

### (c)

Design, build, and train the PINN for this PDE based on the provided skeleton code **PINN_skeleton.py**.

Plot the training error versus training epochs.

---

### (d)

Explain your hyperparameter tuning strategy and comment on the performance of your PINN.

---

### (e)

To improve the performance of the PINN, you are provided with measurement data of the displacement field $u(x)$ at **50 points in the domain**. Incorporate this information into the training process and plot the training error versus training epochs.

---

### (f)

Visualise your results by plotting the tensile stress field $ \sigma_{11}(x) $. Comment on the performance of the trained PINN.

---

**Hint:**
For a PINN to work well, one typically needs a large number of training epochs (on the order of **50–100k**). You may also find that training a PINN using only the PDE residual loss gives unsatisfactory results—this is entirely expected. Adding measurement data can significantly improve convergence toward the reference (ground-truth) solution provided in the MATLAB code.

---

# Problem 2: Learning PDE: 2D Darcy flow (55 marks)

Let us consider learning the solution operator of a **2D Darcy flow problem**.

Consider the following **2-D Darcy Flow equation** on the unit box

$$
-\nabla \cdot (a(x)\nabla u(x)) = f(x), \qquad x \in (0,1)^2
$$

with **Dirichlet boundary condition**

$$
u(x) = 0, \qquad x \in \partial (0,1)^2
$$

where

* $a(x)$ is the diffusion coefficient generated from a random Fourier field
* $f(x)$ is the forcing function which we keep as a constant throughout the domain.

Our aim is to learn the **operator mapping the diffusion coefficient to the solution** defined by

$$
\psi^\dagger : a \mapsto u
$$

using the operator learning methods we discussed in the class.

You are provided a training dataset **Darcy_2D_data_train.mat**, and a test dataset **Darcy_2D_data_test.mat**, which are generated from the MATLAB script **Darcy_2D.m**.

---

### (a)

Design, build and train a **convolutional neural network (CNN)** to learn the solution operator.

Plot

* the train and test losses versus the training epochs
* contour plots for truth and approximate solutions of the solution $u$ for a single sample from the test data

Explain your choice of

* the CNN architecture
* optimizer
* learning rate

You may use the provided Python skeleton code **Darcy_CNN_skeleton.py**.

---

### (b)

Design, build and train a **Fourier Neural Operator (FNO)** architecture to learn the solution operator.

Plot

* the train and test losses versus the training epochs
* contour plots for truth and approximate solutions of $u$ for a single sample from the test dataset

Explain your choice of

* the FNO architecture
* optimizer
* learning rate

You may use the provided Python skeleton code **Darcy_FNO_skeleton.py**.

---

# Submission requirements

Your submission should include:

* the Python script problems 1 and 2:

  * `PINN.py`
  * `PINN_data.py`
  * `Darcy_CNN.py`
  * `Darcy_FNO.py`

* a **PDF file** required to answer parts of problems 1 and 2.

Please note that commenting your code to make it clearly understandable will help the grading process.

Package all of the above in a single **.zip file** and submit it using the **Assignments tool in Moodle**.

---

If you'd like, I can also quickly produce a **cleaner Markdown version designed for Obsidian / GitHub / Notion** (with better figure placeholders and equation formatting), which makes it much easier to tick off tasks while you work through the coursework.
