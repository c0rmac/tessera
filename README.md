# qtt-hank-solver: A Quantized Tensor Train Framework for High-Dimensional Macroeconomic Equilibrium

> **Project Status: Conceptual Proposal** > This repository outlines a technical framework for solving high-dimensional Heterogeneous Agent New Keynesian (HANK) models using Quantized Tensor Trains (QTT).

---

## 1. Problem Statement: The HANK Computational Crisis

Modern macroeconomics is moving away from representative-agent models toward **Heterogeneous Agent (HANK)** frameworks. While more realistic, these models require tracking the joint distribution of wealth, income, and idiosyncratic shocks across millions of agents, leading to a massive state space $\mathbf{x}$.

### 1.1 The Coupled PDE System
A global macroeconomic equilibrium requires finding a stationary value function $V(\mathbf{x})$ and a probability distribution $g(\mathbf{x})$ that satisfy two coupled Partial Differential Equations (PDEs):

1. **Hamilton-Jacobi-Bellman (HJB) Equation**:
    Determines optimal household utility and savings policy:
    
    $$\rho V(\mathbf{x}) = \max_{c} \{u(c) + \mathcal{L}V(\mathbf{x})\}$$
    
    where $\mathcal{L}$ is the infinitesimal generator of the state vector $\mathbf{x}$.

3.  **Kolmogorov Forward Equation (KFE)**:
    Describes the evolution of the household distribution $g(\mathbf{x})$ under the optimal policy:
    
    $$\frac{\partial g}{\partial t} = \mathcal{L}^* g$$
    
    where $\mathcal{L}^*$ is the adjoint operator.

### 1.2 The "Curse of Dimensionality"
In a 10-dimensional problem with a standard grid of $N=100$ points per dimension, the state space explodes to $N^d = 10^{20}$ points. Storing this would require **~400 exabytes** of memory, rendering traditional finite difference schemes and standard Monte Carlo simulations computationally impossible.

---

## 2. Proposed Solution: The Tensorized Economy

This proposal advocates for a deterministic path forward by representing high-dimensional functions entirely within the **Tensor Train (TT)**—or **Matrix Product State (MPS)**—manifold.

### 2.1 Quantized Tensor Trains (QTT)
We propose to "quantize" the state space. By reshaping a dimension with $N$ grid points into $L = \log_2 N$ binary modes, the storage complexity is reduced from $O(N^d)$ to **$O(d \cdot \log N \cdot r^2)$**. 
* **Impact**: This enables the use of hyper-fine grids (e.g., $2^{60}$ points) to capture agent behavior at the extreme tails of the distribution without the exponential memory cost.

### 2.2 Operators as MPOs
The differential operator $\mathcal{L}$ (incorporating drift and diffusion) will be constructed as a **Matrix Product Operator (MPO)**. This allows us to apply the infinitesimal generator to the compressed value function directly in the tensor domain.

### 2.3 Compressed Nonlinearity (Zip-Up Algorithm)
To solve the $\max$ operator in the HJB equation without decompressing the tensors, we propose using the **Zip-Up Algorithm**. By utilizing a smooth Boltzmann-style approximation:

$$\max(A, B) \approx \frac{Ae^{kA} + Be^{kB}}{e^{kA} + e^{kB}}$$

we can perform element-wise maximization with $O(N \cdot r^3)$ scaling, preserving the efficiency of the compressed format.

---

## 3. Technical Strategies for Stability

### 3.1 Taming "Rank Explosion" at Kinks
Economic policy functions often contain "kinks" (non-differentiable points) due to borrowing constraints or tax brackets. These kinks usually cause the tensor rank $r$ to explode.
* **Analytic Smoothing**: We propose replacing non-smooth operators (like ReLU) with the **Softplus** function: $f_\mu(x) = \mu \ln (1 + \exp (x/\mu))$. 
* **Result**: This enforces exponential singular value decay, ensuring the bond dimension remains computationally manageable.

### 3.2 Global General Equilibrium Constraints
Market-clearing conditions (e.g., Aggregate Assets = Aggregate Capital) must be satisfied.
* **U(1) Symmetry**: We intend to use techniques from quantum physics to construct tensors that satisfy conservation laws by construction.
* **Riemannian Optimization**: Optimization gradients will be projected onto the tangent space of the tensor manifold to maintain low-rank structure while satisfying aggregate economic constraints.

---

## 4. Proposed Metrics: Bond Dimension as Systemic Risk

Beyond solving the model, this project proposes using the **Bond Dimension ($r$)** as a novel metric for **Economic Entanglement**:
* **Low Rank ($r \ll \infty$):** Indicates a separable economy where agents' decisions are loosely coupled.
* **Rank Explosion:** Acts as a mathematical early-warning sign of a **Financial Phase Transition**—a crisis where constraints bind simultaneously, creating high global correlation.

---

## 5. Technical Challenges & Research Frontiers

The transition from a theoretical low-rank approximation to a functional HANK solver involves navigating several numerical hurdles. These challenges are categorized into the established foundations of the project and the primary areas of ongoing research.

### Group A: The Solved Foundation
These issues have established mathematical strategies within the proposed QTT-HANK framework:

* **1. The Curse of Dimensionality**: Historically the primary barrier in macroeconomics, this is resolved by the **QTT Format**. By reshaping grids into $d \cdot \log_2 N$ binary modes, we mathematically transform exponential complexity into logarithmic scaling, enabling the use of hyper-fine grids (e.g., $2^{60}$ points).
* **2. Operator Explosion & Rank Growth**: Applying differential operators can cause the tensor rank to inflate ($r_{new} \approx r_{op} \times r_{val}$). This is managed via **TT-Rounding**; after each operation, we perform a Singular Value Decomposition (SVD) to prune the bond dimension and maintain efficiency.
    
* **3. Kinks & Occasionally Binding Constraints**: Non-differentiable features like borrowing limits prevent singular value decay. This is resolved via **Analytic Smoothing (Softplus)**, which "forces" the function to remain low-rank by approximating kinks with smooth, differentiable curves.
    

### Group B: The Research Frontier
These areas represent the "Known Unknowns" where the interaction between economics and tensor algebra is actively being tested:

* **4. Nonlinear Fixed-Point Instability**: Finding a general equilibrium requires prices to clear the market. In a compressed manifold, small numerical "compression noise" can cause the aggregate capital integral to fluctuate, potentially leading to oscillations or divergence in the fixed-point iteration.
    
* **5. Distribution Transport Instability (KFE)**: Enforcing **positivity** ($g(x) \geq 0$) and **normalization** ($\int g = 1$) in the tensor domain is non-trivial. Ensuring the probability distribution doesn't become negative due to approximation errors is a critical stability risk for the Kolmogorov Forward Equation.
* **6. Error Control & Rank Adaptivity**: We are investigating **Dynamic Rank Adaptivity** to intelligently allocate computational resources. The goal is to automatically increase rank only in sensitive regions—such as the extreme tails of the wealth distribution—while keeping it low in flat, less critical areas of the state space.
* **7. Policy Iteration Instability**: Approximation noise in compressed maximization algorithms (like **Zip-Up**) can cause "chattering." This occurs when the solver bounces between suboptimal policies because the compression error is larger than the actual policy improvement step.

---

## 6. Proposed Implementation Roadmap

* **Phase 1: Operator Engineering**: Constructing the infinitesimal generator $\mathcal{L}$ as a Matrix Product Operator (MPO) and validating it against 1D benchmark cases (e.g., Aiyagari-Huggett models).
* **Phase 2: Nonlinear Solver Development**: Implementing the Zip-Up algorithm for element-wise maximization and integrating analytic smoothing to manage bond dimension growth.
* **Phase 3: Global Equilibrium Integration**: Developing the Riemannian optimization routine to enforce market-clearing conditions while maintaining the low-rank manifold.
* **Phase 4: Comparative Benchmarking**: Evaluating performance and precision against Deep BSDE (Neural Network) solvers and traditional Smolyak sparse grid methods.

---

## 7. Mathematical References & Recent Developments

### Foundational Literature (from Proposal)
* **Oseledets, I. V. (2011)**: "Tensor-train decomposition," *SIAM Journal on Scientific Computing*.
* **Dolgov, S. V., & Savostyanov, D. V. (2014)**: "Alternating minimal energy methods for linear systems in higher dimensions," *SIAM Journal on Scientific Computing*.
* **Khoromskij, B. N. (2011)**: "O(d log N) -Quantics Approximation of Functions and Operators in High-Dimensional Applications."

### Relevant Recent Research (2023–2025)
* **Matveev & Smirnov (2024)**: Provides rigorous QTT rank bounds for power-law functions (relevant for CRRA utility).
* **Ye & Loureiro (2023/2024)**: "Quantized tensor networks for solving the Vlasov–Maxwell equations" (provides methods for preserving positivity in KFE-style distributions).
* **Liu, Lee, & Zhang (2024)**: "Tensor Quantile Regression with Low-Rank Tensor Train Estimation" (methods for calibrating distributions to real-world wealth data).
* **ArXiv (2025)**: "Tensor Networks for Liquids in Heterogeneous Systems" (demonstrates QTT superiority in multi-scale heterogeneous environments).
