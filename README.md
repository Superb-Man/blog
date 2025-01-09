# Unveiling the Power of Linear Transformers 

*By **Lara-1905062** and **Toriqe-1905104***  


---

## Introduction  

In the world of machine learning, transformers are the undisputed champions, driving breakthroughs in natural language processing (NLP), computer vision, and beyond. Systems like GPT, BERT, and DALL·E owe their success to this powerful architecture. But there’s a hidden gem in transformers that often gets overlooked: **in-context learning (ICL)**—the ability to solve tasks by interpreting data sequences during inference without retraining.  

While traditional transformers excel at ICL, they come with a heavy computational cost due to their quadratic complexity ($$O(N^2)$$). But, **linear transformers** reduces this complexity to $O(N)$, offering a faster, more scalable alternative. The NeurIPS 2024 paper ["Linear Transformers are Versatile In-Context Learners"](https://neurips.cc) reveals how linear transformers maintain surprising optimization abilities while staying computationally efficient.  

This blog breaks down the paper's insights and explains how these simplified transformers can match the learning power of their more complex counterparts.  

---

## Why Study Linear Transformers?  

### The Complexity Problem  

Traditional transformers handle attention with a quadratic cost: $O(N^2)$, where $N$ is the sequence length. This becomes a bottleneck for applications like processing long documents, genomic sequences, or videos. Imagine processing a sequence of 10,000 tokens:  

- Traditional Transformer: $10,000^2 = 100,000,000$ operations  
- Linear Transformer: $10,000 \times C = 10,000C$ (with $C$ as a small constant)  

This drastic reduction allows linear transformers to process long sequences efficiently, making them ideal for edge devices, real-time applications, and large-scale datasets.  

---

## Transformer vs. Linear Transformer  

### 1. Transformer: Self-Attention Mechanism  
The self-attention mechanism in the original Transformer is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### Intuition and Breakdown:  
- **Step 1: Compute Scores.**  The dot product $QK^\top$ gives similarity scores between the query ($Q$) and key ($K$) embeddings.
- **Step 2: Scale Scores.**  The scores are scaled by $\sqrt{d_k}$ to prevent large values that could saturate the softmax function.
- **Step 3: Normalize Scores.**  A softmax operation ensures the scores sum to 1, making them interpretable as probabilities.
- **Step 4: Aggregate Values.**  The normalized scores are multiplied with the value ($V$) embeddings to produce the weighted sum.

This process computes the attention matrix, which scales quadratically with sequence length $N$, leading to the computational bottleneck for long inputs.

---

### 2. Linear Transformer: Kernelized Self-Attention  
Linear Transformers approximate the self-attention mechanism using kernel functions, bypassing the softmax computation:

$$
\text{Attention}(Q, K, V) = \phi(Q) \left(\phi(K)^\top V\right)
$$

### How It Works:  
- **Step 1: Apply Feature Map.**  The input embeddings $Q$ and $K$ are transformed using a feature map $\phi(\cdot)$.
- **Step 2: Aggregate Values.**  Instead of computing the full attention matrix, we first compute $\phi(K)^\top V$, which scales linearly with $N$. Then, we multiply this result with $\phi(Q)$ to get the final output.
  
By approximating the softmax function with kernel-based transformations, the computation becomes more efficient without significant loss of accuracy. The $N \times N$ matrix multiplication is replaced by two smaller matrix operations, making it $O(N)$.

---
A diagram of Full Transformer and Linear Transformer presented -
![Linear_Attention](https://github.com/lucidrains/linear-attention-transformer/blob/master/linear-attention.png)

---

## What Problems Do Linear Transformers Solve?  

The authors tackled three big questions:  
1. **Can linear transformers handle optimization tasks like full transformers?**  
2. **Are they robust to noise in real-world data?**  
3. **Can they discover new optimization algorithms during training?**  

The answers push linear transformers beyond just being a faster alternative—they’re shown to be smarter too.  

---

## Key Contributions  

### 1. Linear Transformers as Optimizers  

A striking revelation is that **linear transformers inherently act as optimizers**. Each layer mimics a step of preconditioned gradient descent (GD), refining weights iteratively during inference.  

#### Derivation and Simplification:  
Given an input sequence $e_1, e_2, \dots, e_n \in \mathbb{R}^{d+1}$, a single head in a linear self-attention layer computes:

$$
\Delta e_i = W_P \sum_{j=1}^n \langle W_Q e_i, W_K e_j \rangle W_V e_j
$$

This can be re-expressed using pre-computed matrices $P = W_P W_V$ and $Q = W_K^T W_Q$,  
such that $P$ and $Q$ are precomputed. Now, we don't need to compute the dot product of the matrices for each $e_i$.  

For all $e_i$, the precomputed matrices remain the same.  
This is achieved by the assumption of higher-dimensional non-linear kernel to linear equivalency.  

Think of it like this:  
Suppose I have a parabolic function $f(x) = x^2 - 4$, all the negative values of the function determine "NO," and all the non-negative values determine "YES". The linear $x$-axis will effectively classify the function. It's as simple as that.

$$
\Delta e_i = \sum_{j=1}^n (e_j^T Q e_i) P e_j
$$

Expanding further:

$$
\Delta e_i = \sum_{j=1}^n \text{similarity}(e_i, e_j) \times \text{transformation}(e_j)
$$

Where:
- **$similarity(e_i, e_j)$**: Computed via $e_j^T Q e_i$.
- **$transformation(e_j)$**: Performed via $P e_j$.

- **$Q$: Similarity Metric.**  Measures how similar the input $e_i$ is to context embeddings $e_j$.
- **$P$: Transformation.**  Applies learned transformations to aggregate and refine embeddings.

This formulation demonstrates how linear transformers perform efficient computation by reducing matrix dimensions and avoiding explicit softmax calculations.

---

### 2. Handling Noisy Data  

Real-world data often includes noise—sensor glitches, missing entries, or measurement errors. Linear transformers adapt dynamically to minimize the noise impact.  

#### Mathematical Insight:
The model effectively solves noisy regression problems by estimating the optimal weights $w^*$ as:

$$
w^* = (\Sigma + \sigma^2 I)^{-1} \alpha
$$

Where:  
- $\Sigma = \sum_{i=1}^n x_i x_i^T$: Covariance matrix summarizing feature interactions.  
- $\sigma^2$: Noise variance, acting as a regularization term.  
- $\alpha = \sum_{i=1}^n y_i x_i$: Weighted sum of the target outputs $y_i$ and features $x_i$.

#### Simplified Dynamics:
Using this formulation, the weight updates can be expressed as:

$$
w^{l+1} = w^l - \eta \nabla L(w^l) + \gamma w^l
$$

Where:
- $\nabla L(w^l)$: Gradient of the loss function.
- $\eta$: Learning rate.
- $\gamma$: Momentum term to smooth updates.

### Momentum-Like Adjustments:  
Linear transformers introduce a momentum term to stabilize updates and adapt dynamically to noise levels:

$$
w^{l+1} = w^l - \eta \nabla f(w^l) + \gamma w^l
$$

1. The momentum term $\gamma w^l$ helps retain memory of previous updates, reducing oscillations in weight updates.
2. This ensures smoother convergence even when dealing with noisy gradients.

---

### 3. Discovering Optimization Algorithms  

During training, linear transformer runs linear regression on each of its layer. They autonomously discover a momentum-based optimization algorithm resembling Adam or RMSprop.  

#### Advanced Dynamics:
The model refines weights with an adaptive momentum term:

$$
w^{l+1} = w^l - \eta \nabla f(w^l) + \beta (w^l - w^{l-1})
$$

Where:
- $\beta$: Momentum coefficient dynamically adjusted based on past weight changes.


1. The term $\beta (w^l - w^{l-1})$ captures trends in weight updates, ensuring faster convergence.
2. This mimics popular optimizers like Adam but emerges naturally during training.

#### Adaptive Rescaling:
Linear transformers adjust scaling based on noise variance $\sigma^2$:

$$
y^{l+1} = y^l \cdot (1 + \omega \lambda^l)
$$

Where:
- $\lambda^l$: Regularization parameter influenced by noise.
- $\omega$: Scaling factor ensuring stability.

1. The rescaling factor $1 + \omega \lambda^l$ allows the model to dynamically adjust to varying noise levels.
2. This prevents overfitting to noisy inputs and maintains robustness across diverse datasets.

---

## A Deep Diving to the Underlying Maths

### Understanding Recursive Updates in Linear Transformers

In the context of linear transformers, recursive updates form the backbone of parameter optimization. These updates leverage gradient-like descent mechanics to refine weights and matrices iteratively. The key equations in the recursive update framework are:

$$
M^{l+1} = (I + A^l)M^l + b^l(w^l)^T
$$
$$
u^{l+1} = (I + A^l)u^l + a^l b^l
$$
$$
a^{l+1} = (1 + d^l)a^l + (c^l)^T w^l
$$
$$
w^{l+1} = (1 + d^l)w^l - (M^{l+1})^T c^l
$$

1. **Matrix Update ($M^{l+1}$):**
   - This equation updates the matrix $M$, which encapsulates the combined effects of previous layers' parameters ($M^l$), scaling factor ($A^l$), and weighted contributions of $b^l(w^l)^T$.
   - **Intuition**: Think of $M$ as a dynamic memory unit that adjusts to new input features while retaining relevant past information.

2. **Vector Update ($u^{l+1}$):**
   - Updates the vector $u$ using both a scaled version of its prior state and the product $a^l b^l$, which introduces nonlinearity.
   - $u$ acts as a bias tracker, adapting dynamically as $a$ and $b$ evolve.

3. **Scalar Updates ($a^{l+1}, w^{l+1}$):**
   - These updates refine individual weights and biases, incorporating higher-level adjustments from $c^l$ and $d^l$.
   - They ensure that the gradient-like updates balance memory retention with responsiveness to changes in input dynamics.

---

### Power of Diagonal Attention Matrices

To optimize performance and computational efficiency, linear transformers often constrain matrices to be diagonal. This parameterization simplifies the update process significantly. Key matrices $P_k^l$ and $Q_k^l$ are represented as:

$$
P_k^l = \begin{pmatrix} p_{x,k}^l & 0 \newline 0 & p_{y,k}^l \end{pmatrix}, \quad
Q_k^l = \begin{pmatrix} q_{x,k}^l & 0 \newline 0 & q_{y,k}^l \end{pmatrix}.
$$

By reparameterizing the transformer, the update equations for diagonal attention matrices become:

$$
x^{l+1} = x^l + u_{xx}^l \Sigma x^l + u_{xy}^l \Sigma y^l \lambda^l
$$
$$
y^{l+1} = y^l + u_{yx}^l \Sigma x^l + u_{yy}^l \Sigma y^l \lambda^l
$$

1. **Diagonal Parameterization**:
   - By restricting $P_k^l$ and $Q_k^l$ to diagonal forms, we reduce the number of trainable parameters, focusing on key components while minimizing overfitting risks.
   - This simplification allows attention heads to focus more directly on core input features.

2. **Layer Updates ($x^{l+1}, y^{l+1}$):**
   - Updates are computed using learned weights ($u_{xx}^l, u_{xy}^l, u_{yx}^l, u_{yy}^l$) that incorporate attention contributions from both $x^l$ and $y^l$.
   - These updates balance cross-dimensional and self-dimensional information flow for improved learning stability.

---

### Gradient-Like Updates with Diagonal Parameters

In the context of linear transformers, Lemma 4.4 on the original paper introduces updates for $u^{l+1}$ and $w^{l+1}$ in the setup of Theorem 4.1, where diagonal parameterization simplifies computations. These updates are expressed as:

$$
u^{l+1} = (I - \Lambda^l)u^l + \Gamma^l \Sigma \left( a^l w^l - w^l \right)
$$
$$
w^{l+1} = (1 + s^l)w^l - \Pi^l \Sigma \left( a^l w^l - w^l \right) - \Phi^l
$$

Here, $\Lambda^l$, $\Gamma^l$, $s^l$, $\Pi^l$, and $\Phi^l$ are matrices and scalars derived from $M^l$, $u^l$, $a^l$, and $w^l$. 

1. **Vector Update ($u^{l+1}$):**
   - The term $(I - \Lambda^l)u^l$ applies a scaling transformation to the previous vector $u^l$, while $\Gamma^l \Sigma \left( a^l w^l - w^l \right)$ introduces corrections based on the difference between weighted and unweighted states.
   - $u^{l+1}$ evolves by balancing past contributions with dynamic feedback driven by $w^l$ and $a^l$.

2. **Weight Update ($w^{l+1}$):**
   - The term $(1 + s^l)w^l$ amplifies the previous weights, while $\Pi^l \Sigma \left( a^l w^l - w^l \right)$ applies gradient-like corrections, and $\Phi^l$ ensures regularization.
   - $w^{l+1}$ adjusts adaptively, ensuring that weights converge smoothly while retaining critical features of previous layers.

---

### Gradient Descent with Momentum

Lemma 4.4 in paper further relates the updates to a gradient descent approach with momentum, where the function $f(w^l)$ is proportional to the gradient of a linear model:

$$
f(w^l) = \sum_{i=1}^n \left(a^l y_i - \langle w^l, x_i \rangle\right)^2
$$

Using this, the updates can be reinterpreted as:

$$
u^{l+1} = (1 - \beta^l)u^l + \nabla f(w^l)
$$
$$
w^{l+1} = w^l - \eta^l u^l
$$


1. **Gradient Descent with Momentum**:
   - $u^l$ acts as a momentum term, incorporating a weighted history of gradients, which helps accelerate convergence and smooth oscillations.
   - This approach mimics classical momentum in optimization, enhancing stability and performance in high-dimensional parameter spaces.

2. **Weight Refinement**:
   - The term $w^{l+1} = w^l - \eta^l u^l$ refines the weights by stepping in the direction of $-u^l$ scaled by the learning rate $\eta^l$.
   - This ensures that the weights converge to a local minimum of the loss function.

---



## Experiment

### Overview
We evaluated linear transformers on noisy linear regression tasks to test their ability to adapt to varying noise levels and discover in-context learning strategies. Three model types were compared:
- **FULL:** Full parameter matrices.
- **DIAG:** Diagonal parameter matrices.
- **GD++:** A simplified variant with constrained updates.

### Methodology
Models were trained using the Adam optimizer for 200,000 iterations with a learning rate of 0.0001. Noise variances followed uniform or categorical distributions. The **adjusted evaluation loss** (difference between prediction loss and oracle loss) was used as the primary metric.

### Results
- **Uniform Noise:** FULL and DIAG models showed comparable performance, surpassing GD++. Models with 5+ layers matched or exceeded baseline methods like TUNEDRR.
- **Categorical Noise:** DIAG models generalized better to unseen noise levels, while FULL models had lower in-distribution errors but struggled with extrapolation.

### Key Insights
1. Linear transformers effectively adapt to noise levels and learn optimization strategies.
2. DIAG models perform strongly under constrained settings and generalize well.
3. More layers significantly enhance performance across tasks.

These results highlight the potential of linear transformers as efficient and robust in-context learners, even in noisy environments.


### Visualizations
Below are sample performance plots illustrating the adjusted evaluation loss across various noise levels and model configurations:

![Performance_of_models](https://github.com/user-attachments/assets/db373ae1-520c-4255-9826-23d3d87cf5be)

---
## Conclusion
Linear transformers challenge the assumption that model simplicity leads to reduced performance. This paper demonstrates that even reduced-complexity models can exhibit powerful emergent behaviors. By acting as implicit optimizers and discovering new algorithms, linear transformers reinforce their position as viable, efficient alternatives to full transformers.
