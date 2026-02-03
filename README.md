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

## 5. Technical Challenges & Proposed Solutions

The development of the QTT-HANK solver addresses seven core bottlenecks where high-dimensional economic theory meets tensor algebra. Each challenge is paired with a specific deterministic or optimization-based mitigation strategy.

### 1. The Curse of Dimensionality
* **The Problem**: In HANK models with multiple assets and shocks, the state space $\mathbf{x}$ grows exponentially. A 10-dimensional grid with 100 points per axis ($10^{20}$ points) exceeds the global memory capacity of modern supercomputers.
* **The Solution**: **Quantized Tensor Trains (QTT)**. By reshaping the $d$-dimensional grid into $d \cdot \log_2 N$ virtual binary modes, we reduce storage complexity to $O(d \cdot \log N \cdot r^2)$. This enables hyper-fine resolutions ($2^{60}$ points) within a few gigabytes of RAM.

### 2. Operator Explosion & Rank Growth
* **The Problem**: Applying the infinitesimal generator $\mathcal{L}$ (as an MPO) to the value function $V$ (as an MPS) causes the bond dimension to multiply ($r_{new} \approx r_{op} \times r_{val}$), leading to rapid memory exhaustion.
* **The Solution**: **Successive Deterministic Rounding**. After every operator application, we perform a Singular Value Decomposition (SVD)-based truncation. This "re-compression" prunes the redundant information introduced by the operator while maintaining a fixed fidelity threshold.

### 3. Kinks & Occasionally Binding Constraints
* **The Problem**: Borrowing limits and tax brackets introduce non-differentiable "kinks" in policy functions. These kinks break the singular value decay required for low-rank representation, causing "rank explosion."
* **The Solution**: **Analytic Smoothing (Softplus)**. We replace sharp constraints (like ReLU) with a smooth approximation: 
    $$f_\mu(x) = \mu \ln (1 + \exp (x/\mu))$$
    This restores exponential singular value decay and keeps the bond dimension $r$ stable.

### 4. Nonlinear Fixed-Point Instability
* **The Problem**: Solving for General Equilibrium (GE) traditionally requires a hierarchical "Outer Loop" for price discovery (e.g., finding the interest rate $r$) and an "Inner Loop" for the Household HJB/KFE problem. In a QTT framework, aggregate supply and demand curves become "jagged" and non-smooth due to irreducible rounding noise. This makes standard root-finding algorithms like Newton-Raphson or Bisection highly unstable, as the solver frequently gets trapped in local numerical artifacts or diverges when attempting to compute gradients across the compressed tensor landscape.
* **The Solution**: **Simultaneous Stiefel Optimization via Riemannian CBO**.
We collapse the nested hierarchy into a single global energy minimization task. The economic state—comprising the Value Function ($V$), the Distribution ($g$), and the Price vector ($p$)—is optimized as a unified point $\mathcal{X}$ on the **Product Stiefel Manifold** ($St(n,r)^d \times \mathbb{R}^k$). Using the **Riemannian Consensus-Based Optimization (CBO)** framework, a swarm of agents navigates the manifold toward a global equilibrium.

    **The Energy Function**:
    We define the "Economic Energy" $\mathcal{J}(\mathcal{X})$ as a weighted sum of residuals that the CBO swarm aims to minimize:

$$\mathcal{J}(\mathcal{X}) = \underbrace{\|\mathbf{L}_p \mathbf{V} - \mathbf{u}_p\|^2}_{\text{HJB Residual}} + \underbrace{\|\mathbf{L}_p^* \mathbf{g}\|^2}_{\text{KFE Residual}} + \lambda \underbrace{\|\int a g(a,z) da - K(p)\|^2}_{\text{Market Clearing Error}}$$
  
Where $\mathbf{L}_p$ is the infinitesimal generator, $\mathbf{u}_p$ is the utility/return vector, and $\lambda$ acts as the global clearing penalty.



**Parameter Strategy & Quantity Management**:
* **Lambda ($\lambda$) - Penalty Annealing**: We implement a **$\lambda$-schedule** ($\lambda_{t} = \lambda_0 \cdot \gamma^t$, where $\gamma > 1$). By starting with a small $\lambda$, we allow the particles to first explore the space of "rational" household behaviors. As the swarm thermalizes, $\lambda$ is increased to "force" the consensus toward the specific market-clearing price.

### 5. Distribution Transport Instability (KFE)
* **The Problem**: The Kolmogorov Forward Equation (KFE) governs the evolution of the agent distribution $g_t(\mathbf{x})$. In a physically valid economic model, this distribution must satisfy two strict invariants:
    1.  **Positivity**: $g(\mathbf{x}) \geq 0$ for all $\mathbf{x}$ (No negative probabilities).
    2.  **Conservation of Mass**: $\int g(\mathbf{x}) d\mathbf{x} = 1$ (No agent creation/destruction).
    Standard Tensor Train solvers fail these conditions because **SVD Truncation is not positivity-preserving**. "Gibbs oscillations" near sharp cutoffs (like minimum wealth) introduce negative "ghost densities," and repeated rounding operations cause mass leakage ($\int g < 1$), leading to erroneous aggregate capital supplies and interest rate drift.

* **The Solution**: **Wavefunction Squaring (MPS2) on the Spherical TT-Manifold**.
  We abandon the direct simulation of the density $g$. Instead, we represent the distribution as the Born probability amplitude of a latent "wavefunction" tensor $\Psi$:

$$g(\mathbf{x}, t) = |\Psi(\mathbf{x}, t)|^2$$
    
**The Geometric Framework**:
The CBO swarm evolves on the **Spherical Fixed-Rank Manifold** ($\mathcal{S}_{\mathbf{r}}$), defined as the intersection of the Tensor Train manifold and the $L^2$-Unit Sphere:

$$\mathcal{S}_{\mathbf{r}} = \{ \Psi \in \mathcal{M}_{\mathbf{r}} \mid \|\Psi\|_{F} = 1 \}$$
    
**The Algorithm**:
1.  **Tangent Dynamics**: The KFE drift is mapped to the tangent space $T_{\Psi}\mathcal{S}_{\mathbf{r}}$. For a generator $\mathcal{L}^*$, the equivalent evolution for $\Psi$ is:

$$\partial_t \Psi = \frac{1}{2} P_{T_{\Psi}}(\Psi^{-1} \odot \mathcal{L}^*(\Psi \odot \Psi))$$
        
2.  **Spherical Retraction**: After the Consensus Step updates the particle in the tangent space ($\Psi_{tan} = \Psi + \Delta t \cdot \xi$), we apply a **Normalized Retraction**:

$$R_{\Psi}(\xi) = \frac{\text{TT-SVD}(\Psi_{tan})}{\|\text{TT-SVD}(\Psi_{tan})\|_2}$$
    
**Result**: By construction, $g = |\Psi|^2$ is strictly non-negative. By retracting to the sphere, $\int g = \int |\Psi|^2 = 1$ is conserved to machine precision. This ensures the economic model remains physically robust even under aggressive rank compression.

### 6. Error Control & Rank Adaptivity
* **The Problem**: Static bond dimensions either waste VRAM on simple areas of the state space or lose critical detail in complex regions (like the extreme wealth tails).
* **The Solution**: TODO

### 7. Policy Iteration Instability (Chattering)
* **The Problem**: Under compression, the "Zip-Up" maximization step can introduce small errors. If these errors are larger than the improvement gained in the policy step, the solver "chatters"—bouncing between suboptimal policies without converging.
* **The Solution**: TODO

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
