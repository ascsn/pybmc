# Theoretical Background

This section provides a detailed explanation of the theoretical foundations of the `pybmc` package, including Bayesian Model Combination (BMC), Singular Value Decomposition (SVD) for orthogonalization, and the Gibbs sampling methods used for inference.

## Bayesian Model Combination (BMC)

Bayesian Model Combination is a statistical framework for combining predictions from multiple models. Instead of selecting a single "best" model, BMC computes a weighted average of all models, where the weights are determined by the models' performance on the observed data.

### Mathematical Formulation

Given a set of \(K\) models, \(M_1, M_2, \dots, M_K\), the combined prediction for a data point \(x\) is given by:

\[
y(x) = \sum_{k=1}^K w_k f_k(x)
\]

where:
- \(f_k(x)\) is the prediction of model \(M_k\) for input \(x\).
- \(w_k\) is the weight assigned to model \(M_k\), with the constraints that \(\sum_{k=1}^K w_k = 1\) and \(w_k \ge 0\).

In the Bayesian framework, we treat the weights \(\mathbf{w} = (w_1, \dots, w_K)\) as random variables and aim to infer their posterior distribution given the observed data \(D\). Using Bayes' theorem, the posterior distribution is:

\[
p(\mathbf{w} | D) \propto p(D | \mathbf{w}) p(\mathbf{w})
\]

where \(p(D | \mathbf{w})\) is the likelihood of the data given the weights, and \(p(\mathbf{w})\) is the prior distribution of the weights.

## Orthogonalization with Singular Value Decomposition (SVD)

In practice, the predictions from different models are often highly correlated. This collinearity can lead to unstable estimates of the model weights and can cause overfitting. To address this, `pybmc` uses Singular Value Decomposition (SVD) to orthogonalize the model predictions before performing Bayesian inference.

SVD decomposes the matrix of centered model predictions \(X\) into three matrices:

\[
X = U S V^T
\]

where:
- \(U\) is an \(N \times N\) orthogonal matrix whose columns are the left singular vectors.
- \(S\) is an \(N \times K\) rectangular diagonal matrix with the singular values on the diagonal.
- \(V^T\) is a \(K \times K\) orthogonal matrix whose rows are the right singular vectors.

By keeping only the first \(m \ll K\) singular values and vectors, we can create a low-rank approximation of \(X\) that captures the most important variations in the model predictions while filtering out noise and redundancy. This results in a more stable and robust inference process.

## Gibbs Sampling for Inference

`pybmc` uses Gibbs sampling to draw samples from the posterior distribution of the model weights and other parameters. Gibbs sampling is a Markov Chain Monte Carlo (MCMC) algorithm that iteratively samples from the conditional distribution of each parameter given the current values of all other parameters.

### Standard (Unconstrained) Sampler

The unconstrained sampler in `pybmc` assumes the Gaussian likelihood \(y_i \sim \mathcal{N}(X_i \cdot b,\, \sigma_i^2)\) where the per-point variance is linear in a basis of heteroscedasticity metrics, \(\sigma_i^2 = \phi_i \cdot \theta\). Every error model shares this one likelihood: the homoscedastic model is simply the case where the basis \(\phi\) is the constant term alone, so \(\theta = [\sigma^2]\) is a single constant variance. All variance parameters are sampled on the \(\sigma^2\) scale.

The regression coefficients \(b\) are updated with a conjugate (weighted least-squares) Gibbs step; the variance parameters \(\theta\) are updated with a positivity-constrained random-walk Metropolis-Hastings step under Gamma priors. Because \(\theta\) stays strictly positive and the training-time basis is non-negative with a leading constant term, every \(\sigma_i^2\) is positive by construction and no variance flooring is applied during sampling.

### Gibbs Sampler with Simplex Constraints

`pybmc` also provides a Gibbs sampler that enforces simplex constraints on the model weights (i.e., \(\sum w_k = 1\) and \(w_k \ge 0\)). This is achieved by performing a random walk in the space of the transformed parameters and using a Metropolis-Hastings step to accept or reject proposals that fall outside the valid simplex region.

#### When to Use Each Mode

| Mode | Description | Use When |
|------|-------------|----------|
| **Unconstrained** (default) | Weights can take any real value | Maximum flexibility; some models may get negative weights to cancel out biases |
| **Simplex** | Weights satisfy \(w_k \ge 0\) and \(\sum w_k = 1\) | You need interpretable weights that form a proper mixture; predictions should stay within the range of individual models |

#### Simplex Constraint Implementation

The simplex constraint is enforced through a Metropolis-within-Gibbs algorithm. In the SVD-reduced coefficient space, the relationship between the regression coefficients \(\boldsymbol{\beta}\) and the model weights \(\boldsymbol{\omega}\) is:

\[
\omega_k = \sum_{j=1}^m \beta_j \hat{V}_{jk} + \frac{1}{K}
\]

where \(\hat{V}\) contains the (normalized) right singular vectors and \(K\) is the number of models. The term \(\frac{1}{K}\) represents the equal-weight baseline.

At each iteration, the algorithm:

1. **Proposes** a new coefficient vector \(\boldsymbol{\beta}^*\) from a multivariate normal centered on the current value.
2. **Projects** the proposal to weight space via \(\boldsymbol{\omega}^* = \boldsymbol{\beta}^* \hat{V} + \frac{1}{K}\).
3. **Rejects** the proposal if any \(\omega_k^* < 0\) (the sum-to-one constraint is automatically satisfied by the SVD structure and the \(\frac{1}{K}\) offset).
4. **Accepts** valid proposals with the Gaussian likelihood ratio \(\min\!\bigl(1,\; \exp\!\bigl[-\bigl(\mathrm{SSR}(\boldsymbol{\beta}^*) - \mathrm{SSR}(\boldsymbol{\beta})\bigr) / (2\sigma^2)\bigr]\bigr)\), where \(\mathrm{SSR}\) is the residual sum of squares — the same homoscedastic Gaussian likelihood used by the unconstrained sampler.
5. **Samples** the error variance \(\sigma^2\) from its inverse-gamma full conditional (retained samples store \(\sigma^2\), matching the variance-scale parametrization of all other samplers).

The `burn` parameter controls the number of burn-in iterations discarded before collecting samples, and the `stepsize` parameter scales the proposal covariance matrix to tune the acceptance rate.