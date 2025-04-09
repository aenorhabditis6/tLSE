# Optimal Convergence Rates in Nonparametric Learning of SDE Kernels via Tamed Least Squares Estimation

## 1. Introduction

Stochastic differential equations (SDEs) are fundamental tools for modeling complex systems with inherent randomness, from financial markets to biological networks. The ability to infer the underlying dynamics from observed data is a central challenge in many scientific disciplines. This project addresses the nonparametric estimation of drift and diffusion kernels in SDEs, with a particular focus on systems of interacting particles.

We propose to implement and extend the recently developed tamed least squares estimator (tLSE) approach, which has been shown to achieve optimal minimax convergence rates for learning interaction kernels. Our project will bridge theoretical guarantees with practical implementation, demonstrating both the method's efficacy on simulated data and its applicability to real-world systems.

## 2. Mathematical Framework

### 2.1 Problem Formulation

We consider two classes of SDEs:

#### Interacting Particle Systems
For a system of N particles with positions X(t) = (X₁(t),...,Xₙ(t))ᵀ, we have:

$$dX(t) = R_\phi[X(t)]dt + \sigma dB(t)$$

where the interaction operator $R_\phi$ is defined as:

$$R_\phi[X]_i = \frac{1}{N}\sum_{j\neq i}\phi(|X_i - X_j|)\frac{X_i - X_j}{|X_i - X_j|}$$

The unknown function $\phi: \mathbb{R}^+ \rightarrow \mathbb{R}$ is the interaction kernel we aim to estimate.

#### General SDEs with State-Dependent Coefficients
We will also consider general SDEs of the form:

$$dX(t) = b(X(t))dt + \sigma(X(t))dB(t)$$

where $b(x)$ and $\sigma(x)$ are unknown functions to be estimated from trajectory data.

### 2.2 Nonparametric Regression Framework

Given M independent trajectories, we observe samples {(X^m, Y^m)}_{m=1}^M where:

$$Y^m = R_\phi[X^m] + \eta^m$$

For interaction kernels, or:

$$Y^m_k = \frac{X^m_{k+1} - X^m_k}{\Delta t} = b(X^m_k) + \frac{\sigma(X^m_k)}{\sqrt{\Delta t}}\xi^m_k$$

For general SDEs, where $\xi^m_k$ are standard normal random variables.

The estimation problem can be framed as a nonparametric regression, where we aim to minimize the empirical mean-square loss:

$$E_M(\phi) = \sum_{m=1}^M \frac{1}{N}\|Y^m - R_\phi[X^m]\|^2$$

### 2.3 The Tamed Least Squares Estimator

The tLSE approach expands the unknown function in terms of basis functions:

$$\hat{\phi}_{n,M} = \sum_{k=1}^n \hat{\theta}_k \psi_k$$

where the coefficients are computed as:

$$\hat{\theta}_{n,M} = [A^M_n]^{-1}b^M_n \mathbf{1}_{\{\lambda_{\min}(A^M_n) > \frac{1}{4}c_L\}}$$

Here:
- $A^M_n$ is the normal matrix with entries $A^M_n(k,l) = \frac{1}{MN}\sum_{m=1}^M \langle R_{\psi_k}[X^m], R_{\psi_l}[X^m]\rangle$
- $b^M_n$ is the normal vector with entries $b^M_n(k) = \frac{1}{MN}\sum_{m=1}^M \langle R_{\psi_k}[X^m], Y^m\rangle$
- $c_L$ is the coercivity constant
- $\lambda_{\min}(A^M_n)$ is the smallest eigenvalue of the normal matrix

The key innovation of the tLSE is the taming procedure that sets the estimator to zero when the smallest eigenvalue falls below a threshold, addressing the challenge of ill-conditioned normal matrices.

## 3. Proposed Research

### 3.1 Core Methodology

Our central approach will be based on the tamed least squares estimator (tLSE), which has been demonstrated to achieve optimal minimax convergence rates for learning interaction kernels. The tLSE method's key innovation lies in its ability to handle ill-conditioned normal matrices by setting the estimator to zero when the smallest eigenvalue falls below a critical threshold.

We will implement the tLSE with careful attention to:
- Selection of appropriate basis functions for different types of SDEs
- Efficient computation of normal matrices for large datasets
- Optimal threshold selection for the taming procedure
- Adaptive dimension selection based on data characteristics

### 3.2 Theoretical Extensions

We propose to extend the tLSE approach in several directions:

1. **Adaptation to General SDE Settings**: We will adapt the tLSE methodology to the estimation of general state-dependent drift and diffusion coefficients, exploring whether the optimal minimax rates can be maintained.

2. **Analysis of Eigenvalue Structures**: We will investigate the spectral properties of the normal matrix for different types of SDEs, connecting eigenvalue decay patterns to the problem's inherent difficulty.

3. **Theoretical Analysis of Irregular Sampling**: We will extend the convergence analysis to cases where data is sampled at irregular time intervals, which is common in real-world applications.

### 3.3 Implementation and Algorithms

We will develop efficient algorithms for:

1. **Basis Function Selection**: Implementing adaptive selection of basis functions based on data characteristics.

2. **Dimension Selection**: Automatically determining the optimal number of basis functions n ≈ M^(1/(2β+1)) based on sample size and estimated smoothness.

3. **Eigenvalue Computation**: Efficiently computing and monitoring the smallest eigenvalues of the normal matrix for large-scale problems.

### 3.4 Numerical Experiments

We will conduct comprehensive numerical studies on:

1. **Cucker-Smale Flocking Model**: Estimating interaction kernels in simulated flocking systems with varying numbers of particles and noise levels.

2. **Ornstein-Uhlenbeck Processes**: Applying our method to non-interactive SDEs with known analytical solutions to verify accuracy.

3. **Discontinuous Kernels**: Testing the method's performance on piecewise constant interaction functions, leveraging the ability of tLSE to handle β ≤ 1/2 cases.

## 4. Evaluation Metrics and Validation

We will assess the performance of our methods using multiple criteria:

### 4.1 Convergence Rate Verification

The primary metric will be the L² error between estimated and true kernels:

$$\|\hat{\phi} - \phi^*\|^2_{L^2_\rho} = \int |\hat{\phi}(r) - \phi^*(r)|^2 \rho(r) dr$$

where ρ is the exploration measure (distribution of pairwise distances).

We will analyze:
- Log-log plots of error vs. sample size M to verify the slope matches -2β/(2β+1)
- Comparison of empirical rates with theoretical predictions across different function classes

### 4.2 Minimax Optimality Testing

To verify the minimax optimality, we will:
- Construct 2s-separated hypothesis functions as described in the theoretical framework
- Compute errors for various estimation methods (not just tLSE)
- Verify that no method achieves better worst-case performance than predicted by theory

### 4.3 Comparative Analysis with Baseline Methods

We will benchmark the tLSE against several established baseline methods for nonparametric estimation:

1. **Standard Least Squares Estimator (LSE)**: The conventional approach without taming, implemented with Moore-Penrose pseudo-inverse to handle ill-conditioned matrices.

2. **Tikhonov Regularization**: Adding a regularization term λ||φ||² to the objective function to stabilize estimation in the presence of small eigenvalues.

3. **Truncated SVD Estimator**: Using singular value decomposition with truncation of small singular values.

4. **Kernel Ridge Regression**: Employing kernel methods with appropriate radial basis functions.

5. **Gaussian Process Regression**: Utilizing Bayesian nonparametric approaches with carefully selected priors.

For each method, we will analyze:
- Convergence rates under different smoothness conditions
- Robustness to ill-conditioning 
- Computational efficiency
- Prediction accuracy on test trajectories

### 4.4 Practical Performance Metrics

Beyond theoretical validation, we will assess practical performance through:
- Trajectory prediction accuracy using estimated kernels
- Computational efficiency (time and memory requirements)
- Robustness to noise and parameter variations
- Stability of the estimation procedure across multiple runs

## 5. Expected Innovations and Contributions

Our project will make several significant contributions:

1. **Practical Implementation of Theoretical Results**: Translating the abstract tLSE framework into concrete algorithms with demonstrated performance guarantees.

2. **Extended Methodology**: Adapting the tLSE approach to a broader class of SDEs beyond interaction kernels.

3. **Empirical Verification of Minimax Rates**: Providing comprehensive empirical evidence for the theoretical convergence rates.

4. **Software Development**: Producing an open-source implementation that enables researchers to apply these methods to their own datasets.

5. **Visualization Tools**: Developing novel visualization approaches for understanding the behavior of tLSE in high-dimensional spaces.

## 7. Conclusion

This project aims to bridge theory and practice in nonparametric estimation of SDE kernels. By implementing and extending the tLSE methodology, we will demonstrate its effectiveness for learning both interaction kernels in particle systems and general state-dependent coefficients in SDEs. Our work will validate theoretical convergence rates empirically and develop practical tools for researchers working with stochastic dynamical systems.

The resulting methods will have applications in diverse fields including financial mathematics, molecular dynamics, collective behavior modeling, and systems biology. The theoretical insights gained may also contribute to the broader understanding of minimax rates in nonparametric regression problems with nonlocal dependencies.
