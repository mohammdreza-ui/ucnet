# Mathematical Foundations & Derivations in UCNet

This document details the rigorous mathematical derivations underlying the UCNet automatic differentiation engine, neural network layers, optimization algorithms, and uncertainty quantification metrics.

---

## 1. Reverse-Mode Automatic Differentiation

### 1.1 Computational Graph and Adjoints
Let $L \in \mathbb{R}$ denote a scalar loss objective computed through a directed acyclic graph (DAG) of tensor operations. For any intermediate variable (tensor node) $x$, we define its **adjoint** (sensitivity) as:

$$\bar{x} \equiv \frac{\partial L}{\partial x}$$

For an operation $y = f(x)$, by the vector chain rule:

$$\bar{x} = \left(\frac{\partial y}{\partial x}\right)^{\!\top} \bar{y}$$

### 1.2 Multi-Consumer Accumulation (Branching in DAG)
If a tensor $x$ is consumed by multiple downstream operations $y_1 = f_1(x), y_2 = f_2(x), \dots, y_K = f_K(x)$, the total variation in $L$ with respect to $x$ is the sum of partial variations through each path:

$$\bar{x} = \sum_{k=1}^K \left(\frac{\partial y_k}{\partial x}\right)^{\!\top} \bar{y}_k$$

This explains why reverse-mode autodiff implementations must **accumulate** gradients via `+=` rather than overwriting (`=`):
```python
x.grad = g if x.grad is None else x.grad + g
```

### 1.3 Why Reverse-Mode?
For machine learning objectives, $L: \mathbb{R}^P \to \mathbb{R}$ maps $P$ parameters (where $P$ can be millions) to a single scalar loss.
- **Forward-mode autodiff** computes directional derivatives $\frac{\partial y}{\partial x_i}$ column-by-column, requiring $\mathcal{O}(P)$ passes through the computational graph.
- **Reverse-mode autodiff** propagates sensitivities backwards from $L$, computing the entire gradient vector $\nabla_\theta L \in \mathbb{R}^P$ in a single backward pass whose computational complexity is proportional to a constant multiple ($\approx 2\times$ to $3\times$) of the forward evaluation cost, completely independent of the parameter dimension $P$.

---

## 2. Matrix Multiplication (`matmul`) Adjoint Derivation

Let $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, and define $C = AB \in \mathbb{R}^{m \times n}$.

### 2.1 Matrix Differentials and Trace Inner Product
The Frobenius inner product of matrices $X, Y \in \mathbb{R}^{m \times n}$ is given by $\langle X, Y \rangle = \operatorname{tr}(X^\top Y)$. The total differential of the scalar loss $L$ is:

$$dL = \operatorname{tr}(\bar{C}^\top dC)$$

Taking the differential of $C = AB$:

$$dC = (dA) B + A (dB)$$

Substitute $dC$ into $dL$:

$$dL = \operatorname{tr}\!\big(\bar{C}^\top (dA) B\big) + \operatorname{tr}\!\big(\bar{C}^\top A (dB)\big)$$

### 2.2 Trace Cycling Property
Using the cyclic permutation property of the trace ($\operatorname{tr}(XYZ) = \operatorname{tr}(ZXY) = \operatorname{tr}(YZX)$):

$$\operatorname{tr}\!\big(\bar{C}^\top (dA) B\big) = \operatorname{tr}\!\big(B \bar{C}^\top dA\big) = \operatorname{tr}\!\big((\bar{C} B^\top)^\top dA\big)$$

$$\operatorname{tr}\!\big(\bar{C}^\top A (dB)\big) = \operatorname{tr}\!\big((A^\top \bar{C})^\top dB\big)$$

Equating with $dL = \operatorname{tr}(\bar{A}^\top dA) + \operatorname{tr}(\bar{B}^\top dB)$ yields the exact analytical adjoints:

$$\boxed{\bar{A} = \bar{C} B^\top, \qquad \bar{B} = A^\top \bar{C}}$$

For batched tensors $(\dots, m, k) \times (\dots, k, n) \to (\dots, m, n)$, transposition is applied strictly to the last two axes (`swapaxes(-1, -2)`), followed by gradient unbroadcasting.

---

## 3. Multidimensional Gradient Unbroadcasting

When an array $A$ of shape $S_A$ is broadcast to shape $S_C$ during an elementwise operation, each element of $A$ contributes to multiple elements in $C$. In the backward pass, the adjoint $\bar{C}$ must be summed along all broadcasted axes:

1. **Leading Dimensions:** If $\operatorname{ndim}(\bar{C}) > \operatorname{ndim}(A)$, sum over the first $\operatorname{ndim}(\bar{C}) - \operatorname{ndim}(A)$ axes.
2. **Size-1 Dimensions:** For any axis where the target dimension is $1$ and the broadcasted dimension is $> 1$, sum along that axis with `keepdims=True`.
3. **Scalar Target:** If $S_A = ()$, sum over all elements: $\sum \bar{C}$.

---

## 4. 1D Convolution (`Conv1D`) Adjoint Derivation

Let input $x \in \mathbb{R}^{N \times C_{in} \times L}$, filter weights $w \in \mathbb{R}^{C_{out} \times C_{in} \times K}$, bias $b \in \mathbb{R}^{C_{out}}$, stride $s$, and padding $p$. Output length is:

$$T = \Big\lfloor \frac{L + 2p - K}{s} \Big\rfloor + 1$$

The forward correlation output at sample $n$, channel $o$, and temporal index $t$ is:

$$y[n, o, t] = b[o] + \sum_{c=1}^{C_{in}} \sum_{k=0}^{K-1} w[o, c, k] \cdot x_{\text{pad}}[n, c, t \cdot s + k]$$

### Backward Pass:
1. **Bias gradient:**
   $$\bar{b}[o] = \sum_{n=1}^N \sum_{t=0}^{T-1} \bar{y}[n, o, t]$$

2. **Weight gradient:**
   $$\bar{w}[o, c, k] = \sum_{n=1}^N \sum_{t=0}^{T-1} \bar{y}[n, o, t] \cdot x_{\text{pad}}[n, c, t \cdot s + k]$$
   Evaluated efficiently using NumPy `einsum`:
   ```python
   gw[:, :, k] = np.einsum("not,nct->oc", ybar, x_pad[:, :, k : k + s * T : s])
   ```

3. **Input gradient:**
   $$\bar{x}_{\text{pad}}[n, c, t \cdot s + k] \mathrel{+}= \sum_{o=1}^{C_{out}} \bar{y}[n, o, t] \cdot w[o, c, k]$$
   Unpadded via slicing: $\bar{x} = \bar{x}_{\text{pad}}[:, :, p : L_{\text{pad}} - p]$.

---

## 5. Layer Normalization (`LayerNorm`) Adjoint Derivation

For an activation vector $x \in \mathbb{R}^D$:

$$\mu = \frac{1}{D}\sum_{j=1}^D x_j, \qquad \sigma^2 = \frac{1}{D}\sum_{j=1}^D (x_j - \mu)^2$$

$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad y_j = \gamma_j \hat{x}_j + \beta_j$$

Let $g_j = \bar{y}_j \gamma_j$ and $s = \sqrt{\sigma^2 + \varepsilon}$. Because $\hat{x}_j$ depends on $x_j$ through three distinct paths (direct numerator, mean $\mu$, and variance $\sigma^2$):

$$\frac{\partial \hat{x}_k}{\partial x_j} = \frac{1}{s}\left( \delta_{jk} - \frac{1}{D} - \frac{\hat{x}_k \hat{x}_j}{D} \right)$$

Applying the chain rule $\bar{x}_j = \sum_k g_k \frac{\partial \hat{x}_k}{\partial x_j}$:

$$\bar{x} = \frac{1}{s}\Big( g - \bar{g} - \hat{x} \cdot \overline{g \odot \hat{x}} \Big)$$

where $\bar{g} = \frac{1}{D} \sum_j g_j$ and $\overline{g \odot \hat{x}} = \frac{1}{D} \sum_j g_j \hat{x}_j$.

---

## 6. Numerical Stability in Activations

### 6.1 Stable Sigmoid
$$\sigma(a) = \begin{cases} \frac{1}{1 + e^{-a}} & a \ge 0 \\ \frac{e^a}{1 + e^a} & a < 0 \end{cases}$$
Prevents numerical overflow when evaluating $e^{-a}$ for large negative $a$.

### 6.2 Stable Softplus
$$\operatorname{softplus}(a) = \ln(1 + e^a) = \max(a, 0) + \ln(1 + e^{-|a|})$$
Eliminates intermediate overflow for large positive inputs while remaining accurate for small negative inputs.

### 6.3 LogSumExp
$$\operatorname{LSE}(z) = m + \ln\sum_j e^{z_j - m}, \qquad m = \max_j z_j$$
With analytical derivative $\frac{\partial \operatorname{LSE}}{\partial z_j} = \operatorname{softmax}(z)_j$.

---

## 7. Optimization Algorithms

### 7.1 Adam Bias Correction Derivation
The moving average updates are:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t - i} g_i$$

Taking the expectation under the assumption that gradients $g_i$ come from a stationary distribution with mean $\mathbb{E}[g]$:

$$\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t - i} g_i\right] = \mathbb{E}[g] (1 - \beta_1) \sum_{k=0}^{t-1} \beta_1^k = \mathbb{E}[g] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = (1 - \beta_1^t) \mathbb{E}[g]$$

Therefore, dividing by $(1 - \beta_1^t)$ provides an exactly unbiased estimator $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$.

### 7.2 Decoupled Weight Decay (AdamW)
Standard L2 regularization adds $\lambda \theta$ directly to $g_t$, which gets rescaled by $\sqrt{\hat{v}_t} + \varepsilon$, causing parameters with large or frequent gradients to be regularized less than parameters with sparse gradients.
AdamW separates weight decay from the gradient update:

$$\theta_t = \theta_{t-1} (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

---

## 8. Uncertainty Quantification & Calibration

### 8.1 Heteroscedastic Gaussian NLL
Predicts mean $\mu(x)$ and log-variance $s(x) = \ln \sigma^2(x)$:

$$\mathcal{L}(\mu, s; y) = \frac{1}{2} s + \frac{1}{2} (y - \mu)^2 e^{-s} + \frac{1}{2} \ln(2\pi)$$

Adjoints:
$$\frac{\partial \mathcal{L}}{\partial \mu} = -(y - \mu) e^{-s}, \qquad \frac{\partial \mathcal{L}}{\partial s} = \frac{1}{2} - \frac{1}{2} (y - \mu)^2 e^{-s}$$

At stationary points w.r.t $s$:
$$e^s = (y - \mu)^2 \implies \sigma^2(x) = (y - \mu(x))^2$$
The network naturally learns to output the local variance of residual prediction errors (aleatoric uncertainty).

### 8.2 Law of Total Variance (Ensemble Decomposition)
For an ensemble mixture distribution $p(y \mid x) = \frac{1}{M} \sum_{m=1}^M \mathcal{N}(y; \mu_m(x), \sigma_m^2(x))$:

By the Law of Total Variance:
$$\operatorname{Var}[Y \mid x] = \mathbb{E}_{m}[\operatorname{Var}[Y \mid x, m]] + \operatorname{Var}_{m}[\mathbb{E}[Y \mid x, m]]$$

$$\sigma_{\text{total}}^2(x) = \underbrace{\frac{1}{M}\sum_{m=1}^M \sigma_m^2(x)}_{\text{Aleatoric Uncertainty (Expected Noise)}} + \underbrace{\frac{1}{M}\sum_{m=1}^M (\mu_m(x) - \mu^\star(x))^2}_{\text{Epistemic Uncertainty (Model Disagreement)}}$$

where $\mu^\star(x) = \frac{1}{M}\sum_m \mu_m(x)$.

Algebraic identity:
$$\frac{1}{M}\sum_m (\mu_m - \mu^\star)^2 = \frac{1}{M}\sum_m \mu_m^2 - \mu^{\star 2}$$

### 8.3 Closed-Form Gaussian CRPS
The Continuous Ranked Probability Score measures the probabilistic distance between forecast distribution $F$ and target observation $y$:

$$\operatorname{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(t) - \mathbf{1}[t \ge y])^2 dt$$

For a Gaussian forecast $F = \mathcal{N}(\mu, \sigma^2)$:

$$\boxed{\operatorname{CRPS}(\mathcal{N}(\mu, \sigma^2), y) = \sigma \left[ z \big(2\Phi(z) - 1\big) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]}$$

where $z = \frac{y - \mu}{\sigma}$, $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$, and $\Phi(z)$ is the standard normal cumulative distribution function.
