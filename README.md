# Project Tessera

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

`Tessera` is a novel, quantum-inspired solver for **constrained thematic portfolio optimization**, formulated as an **NP-hard** combinatorial problem.

This project reframes the financial optimization problem as a physics **ground state search** of an Ising model Hamiltonian. We employ a **Density Matrix Renormalization Group (DMRG)** algorithm to find the optimal portfolio. The core innovation lies in encoding complex, non-linear constraints **directly into the tensor structure of the Matrix Product Operator (MPO)**, which represents the problem's Hamiltonian. This "constraint-intrinsic" method avoids the instability of traditional penalty-based approaches.

---

## The Problem: NP-Hard Portfolio Construction

A modern portfolio optimization problem goes far beyond the classic Markowitz quadratic program. Let $w \in \mathbb{R}^N$ be the vector of asset weights, $\lambda$ the risk aversion paramter and $r$ the expected return of each asset. The problem must also incorporate a vector of binary decision variables $z \in \{0, 1\}^N$, where $z_i = 1$ if asset $i$ is included in the portfolio and $0$ otherwise.

The objective is to minimize a risk-adjusted objective function $f(w, z)$ subject to a set of complex constraints:

$$
\min_{w, z} \quad f(w, z) = w^T \Sigma w - \lambda r^T w
$$
$$
\text{subject to:}
$$
$$
\begin{aligned}
\mathbf{1}^T w &= 1 && \text{(Full investment)} \\
w_i &\le z_i && \text{(Assets can only have weight if selected)} \\
w_i &\ge 0 && \text{(Long-only)} \\
\quad z_i &\in \{0, 1\} && \text{(Binary selection)} \\
\sum_{i=1}^N z_i &\le K && \text{(Cardinality constraint)} \\
g(w, z) &\le C && \text{(e.g., ESG / sector / factor constraints)}
\end{aligned}
$$

The presence of the binary variables $z$ and the quadratic objective $w^T \Sigma w$ makes this a **Mixed-Integer Quadratic Program (MIQP)**, a well-known NP-hard problem.

---

## Current Solutions & Their Mathematical Limitations

1.  **Heuristic & Metaheuristic Algorithms:**
    * These methods (e.g., Simulated Annealing, Genetic Algorithms) transform the constrained problem into an unconstrained one using **penalty functions**.
    * The objective becomes an unstable penalty-based function $\mathcal{L}(w, z, \lambda)$:
        
        $$\min_{w, z} \quad \mathcal{L} = f(w, z) + \lambda_1 \max\left(0, \sum_i z_i - K\right) + \lambda_2 \max(0, g(w, z) - C)$$
        
    * **Limitation:** The solution's quality and validity are highly sensitive to the hyperparameters $\lambda_i$. If $\lambda_i$ is too small, the solver will "cheat" and return an invalid solution. If $\lambda_i$ is too large, it creates a "rugged" optimization landscape, trapping the solver in a sub-optimal local minimum.

2.  **Classical Exact Solvers (MIQP Solvers):**
    * These solvers (e.g., Gurobi, CPLEX) use methods like **branch-and-bound**.
    * **Limitation:** Their worst-case runtime complexity is exponential, $O(2^N)$, making them computationally intractable for the large asset universes ($N \sim 1000s$) required in real-world finance.

---

## The *Tessera* Approach: Our Solution

`Tessera` maps this problem to a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, which is mathematically equivalent to finding the ground state of an Ising model Hamiltonian.

### 1. Hamiltonian Formulation

We discretize the weights $w_i$ and formulate the entire MIQP as an Ising Hamiltonian $H$, whose operators act on a chain of $N$ "spins" (assets). The goal is to find the lowest-energy configuration, or "ground state" $|\psi_{gs}\rangle$, which corresponds to the optimal portfolio:

$$
|\psi_{gs}\rangle = \underset{|\psi\rangle}{\text{argmin}} \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}
$$

### 2. Constraint-Aware Matrix Product Operator (MPO)

This is the core technical innovation. The Hamiltonian $H$ is not built naively; it is constructed as a **Matrix Product Operator (MPO)** that *natively encodes the constraints*.

An MPO is a tensor-train decomposition of the operator $H$:

$$
H_{(i_1 \dots i_N)}^{(j_1 \dots j_N)} = \sum_{\alpha_0 \dots \alpha_N} A_1[i_1, j_1]_{\alpha_0, \alpha_1} A_2[i_2, j_2]_{\alpha_1, \alpha_2} \dots A_N[i_N, j_N]_{\alpha_{N-1}, \alpha_N}
$$

To enforce a constraint like cardinality ($\sum z_i \le K$), we augment the "virtual" bond indices $\alpha$ to include an auxiliary "counter" state. The local MPO tensor $A_k$ at site $k$ (asset $k$) is structured as a transition matrix on this counter.

Let $\alpha_k = (m_k, n_k)$ where $m_k$ is the standard bond index and $n_k$ is the cumulative asset count $\sum_{i=1}^k z_i$. The MPO tensor $A_k$ is constructed to enforce the logic:

$$
(A_k)_{\alpha_{k-1}, \alpha_k} = 0 \quad \text{if} \quad n_{k-1} + z_k > K
$$

This gives the MPO a block-triangular structure in the auxiliary constraint space. Any state $|\psi\rangle$ that violates the constraint (e.g., has $n_N > K$) is automatically projected out, resulting in $\langle\psi|H|\psi\rangle \to \infty$. The solver is therefore **guaranteed by construction** to only search within the valid, constraint-satisfying Hilbert subspace.

### 3. Density Matrix Renormalization Group (DMRG) Solver

We use the **DMRG algorithm** to find the ground state of this constraint-aware MPO. DMRG is a variational algorithm that iteratively optimizes a **Matrix Product State (MPS)** $|\psi\rangle$ to find the minimal energy $E = \langle \psi | H | \psi \rangle$.

$$
|\psi\rangle = \sum_{i_1 \dots i_N} \left( M_1[i_1] M_2[i_2] \dots M_N[i_N] \right) |i_1 \dots i_N\rangle
$$

The DMRG algorithm is exceptionally efficient at this task because it operates in the low-entanglement corner of the state space, which is precisely where the solutions to many combinatorial optimization problems are known to lie.

This method is **stable, principled, and requires no penalty-tuning**, providing a robust and scalable solver for complex, real-world portfolio optimization.

---

## Limitations and Risk Factors

### 1. Theoretical Limitations: The "Mapping" Problem

The entire premise relies on successfully mapping a complex, continuous-variable financial problem onto a discrete, one-dimensional quantum spin model. This abstraction is the first and most significant hurdle.

#### Failure Point: Discretization of Weights ($w_i$)

The portfolio requires continuous weights $w \in \mathbb{R}^N$, but an Ising model Hamiltonian operates on discrete spins ($z_i \in \{0, 1\}$). To solve this, we must discretize the continuous weights, which introduces two major problems:

1.  **Imprecision:** The true optimal weight for an asset might be $w_i = 0.0531...$, but the nearest discrete representation might be $0.05$. This "discretization error" can lead to a final portfolio that is significantly sub-optimal in the real, continuous-variable world.
2.  **State-Space Explosion:** To achieve acceptable precision, each asset $z_i$ must be represented by multiple spins. For example, using $b=8$ spins (bits) per asset to represent $2^8 = 256$ weight levels. The computational cost of DMRG scales polynomially (e.g., $O(d^3)$) with this local "physical dimension" $d$. A system with $d=256$ is computationally vast and may be intractable, even if the number of assets $N$ is modest.

#### Failure Point: Mapping Complex Constraints

The project's core innovation is encoding constraints directly into the MPO. The proposal for a simple cardinality constraint ($\sum z_i \le K$) using a "counter" index is elegant and feasible.

However, real-world financial constraints are far more complex, often **non-linear and non-local**. Consider constraints like:
* "The total portfolio volatility must be $\le 15\%$."
* "The weighted-average ESG score must be $\ge 7.5$."
* "The total volatility contribution from the 'Technology' sector must be $\le 5\%$."

These constraints depend on global, all-to-all interactions ($w_i, w_j, \Sigma_{ij}$). It may be **mathematically impossible** to formulate such complex, non-local constraints as a finite-state machine that can be compiled into the local tensors of a 1D MPO. If the solver can only handle simple, local constraints, it fails to solve the "holistic" problem it targets.

---

### 2. Computational Limitations: The "Explosion" Problem

Even if the mapping is theoretically sound, the computational resources required may make it fail in practice. The performance of all tensor network methods is governed by the **bond dimension ($\chi$)**, which can "explode."

#### Failure Point: MPO Bond Dimension Explosion

The "constraint-aware" MPO works by adding auxiliary indices to the virtual bond $\alpha$. This is the project's single greatest technical risk.

* A cardinality constraint ($\le K$ assets) requires an auxiliary bond dimension of $\chi_{\text{card}} \approx K+1$.
* A sector-weight constraint (e.g., $\le 10\%$ in Sector A, discretized to 20 bins) requires $\chi_{\text{sector}} \approx 20$.
* If we have 5 such sector constraints, the total auxiliary bond dimension required to track all constraints simultaneously could be the *product* of these dimensions: $\chi_{\text{aux}} \approx \chi_{\text{card}} \times (\chi_{\text{sector}})^5$.

This bond dimension can grow **exponentially with the number of constraints**. The MPO itself could become so large that it cannot be constructed or stored in memory, making the problem unsolvable before DMRG even begins.

#### Failure Point: MPS Bond Dimension Explosion (High Entanglement)

The DMRG algorithm is efficient *only if* the solution (the ground state MPS) is "lowly entangled" and can be accurately represented with a small bond dimension, $\chi_{\text{sol}}$.

* This "low entanglement" assumption (known as an "area law") comes from 1D physical systems where interactions are **local** (each spin only talks to its neighbors).
* A financial portfolio is the *exact opposite*: it is a **globally-coupled, all-to-all** system. The covariance matrix $\Sigma$ links every single asset to every other asset.
* This global coupling will almost certainly produce a **highly entangled** ground state.
* Accurately representing this state will require an MPS bond dimension $\chi_{\text{sol}}$ that may grow exponentially with $N$. Since the runtime of DMRG scales as $O(N \cdot d^3 \cdot \chi_{\text{sol}}^3)$, an exploding $\chi_{\text{sol}}$ makes the algorithm intractably slow.

---

### 3. Practical Limitations: The "Performance" Problem

Finally, even if the project is theoretically sound and computationally feasible, it could fail by simply not being better than mature, existing solutions.

#### Failure Point: DMRG is a Heuristic, Not an Exact Solver

The proposal positions DMRG as a "principled" alternative to "unstable" heuristics. This is true, but DMRG is *also* a heuristic.

It is a **variational algorithm**, meaning it iteratively improves a candidate solution (the MPS). It is **not guaranteed** to find the true, global ground state. For a complex, "frustrated" Hamiltonian (which a financial problem will be), DMRG can get stuck in a local minimum (a sub-optimal portfolio), just as a genetic algorithm can.

#### Failure Point: Slower or Less Accurate than
Classical Solvers

The project's ultimate success is not just "does it run?" but "is it *better*?". It could fail if:

1.  **For Small-to-Medium Problems ($N \sim 500$):** A commercial MIQP solver (like Gurobi or CPLEX) finds the *provably* optimal, *continuous-weight* solution in less time.
2.  **For Large Problems ($N \sim 5000$):** A well-tuned, standard heuristic (like Simulated Annealing or a Genetic Algorithm) running on a GPU finds an equally good (or better) solution in a fraction of the time.

If `Tessera` cannot outperform these benchmarks, it is a sophisticated solution to a problem that is already solved more effectively by other means.

---

## Why 'Tessera'?

The name **Tessera** is a metaphor for the combinatorial optimization problem at the heart of this project.

A "tessera" is a single, small tile used to create a mosaic. By itself, one tile has little value or meaning. However, when you combine thousands of individual tesserae, adhering to a specific set of rules and constraints (the design), you create a large, complex, and optimized image.

This directly mirrors our problem:
* **A Tessera (tile)** = A single asset ($z_i$).
* **The Rules (design)** = The financial constraints ($\sum z_i \le K$, sector limits, etc.).
* **The Mosaic (final image)** = The optimal, fully-constructed portfolio.

`Tessera` is the engine that intelligently selects and places each individual asset to construct this optimal portfolio, perfectly adhering to all constraints.
