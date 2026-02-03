# QTT-HANK: Deterministic High-Dimensional Equilibrium Solver

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

## 5. Potential Challenges & Implementation Risks

Despite the theoretical advantages of the QTT approach, several technical risks remain:

* **Entanglement Scaling (Volume Law Risk)**: QTT efficiency relies on the "Area Law" of entanglement. If the economy enters a regime of extreme high-entropy correlations—where every state variable is strongly coupled to every other—the bond dimension $r$ may scale volumetrically, causing the computational cost to revert to the curse of dimensionality.
* **Persistent Rank Explosion at Kinks**: While analytic smoothing (Softplus) mitigates rank growth, extreme discontinuities in policy functions (e.g., discrete labor choices or complex tax brackets) may still lead to "rank-heavy" cores that exceed available VRAM/DRAM.
* **Numerical Convergence of AMEn**: The **Alternating Minimal Energy (AMEn)** algorithm is highly efficient for linear systems, but its stability in solving the highly non-linear, coupled HJB-KFE system is still being verified. There is a risk of the solver becoming trapped in local minima during the policy iteration phase.

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
