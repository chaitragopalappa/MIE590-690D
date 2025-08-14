# Chapter 13: Neural Networks for Structured (Tabular) Data

## Feed-Forward Neural Network (FFNN) / Multi-Layer Perceptron (MLP)

### Outline
- Objective function (minimize loss)
- Model architecture (multiple layers — *deep*; each node in each layer is a linear transformation of the previous layer passed through an activation function)
- Optimizer (backpropagation with SGD/variants in a directed acyclic graph)

# Multi-Layer Perceptron (MLP)

## Objective function

Suppose $\mathbf{y} = f(\mathbf{x});\ \mathbf{x}\in \mathbb{R}^n,\ \mathbf{y}\in \mathbb{R}^m$.  
We fit the model by minimizing a loss:
$$
\min_{\boldsymbol{\theta}} \ \mathcal{L}(\boldsymbol{\theta}) \;=\; \min_{\boldsymbol{\theta}} \left\|\hat{\mathbf y}-\mathbf y\right\|,
$$
where
- $\hat{\mathbf y}$ are the predictions from the neural network,
- $\mathcal{L}$ is the loss,
- $\boldsymbol{\theta} = \{\,\mathbf b, \mathbf W_1, \mathbf W_2, \dots, \mathbf W_L\,\}$ are the network parameters.

Thus, **model fitting** for an NN means learning $\boldsymbol{\theta}$ by **minimizing** the loss.

<table>
<tr>
<td style="width:50%; vertical-align:top;">

## NN Model Architecture

### Composition
$$
z_L = f_L(z_{L-1}) = f_L\!\big(f_{L-1}(z_{L-2})\big) = \cdots = \big(f_L \circ f_{L-1} \circ \cdots \circ f_1\big)(z_0).
$$

We set $\mathbf x \triangleq z_0$ and $\hat{\mathbf y} \triangleq z_L$.

A typical layer map is
$$
\boldsymbol z_\ell \;=\; f_\ell(\boldsymbol z_{\ell-1})
\;=\; \varphi_\ell\!\left(\mathbf b_\ell + \mathbf W_\ell \boldsymbol z_{\ell-1}\right) \quad\text{(vector form)}.
$$

In scalar form:
$$
z_{k\ell} \;=\; \varphi_\ell\!\left(b_{k\ell} + \sum_{j=1}^{K_{\ell-1}} W_{jk\ell}\, z_{j,\,\ell-1}\right).
$$
If we **augment** $\boldsymbol z_{\ell-1}$ with a leading $1$ (i.e., set $z_{0,\,\ell-1}=1$) and define $W_{0k\ell}=b_{k\ell}$,
this simplifies to
$$
z_{k\ell} \;=\; \varphi_\ell\!\left(\sum_{j=0}^{K_{\ell-1}} W_{jk\ell}\, z_{j,\,\ell-1}\right).
$$
In vector form with the augmentation implied:
$$
\boldsymbol z_\ell \;=\; \varphi_\ell\!\left(\mathbf W_\ell \boldsymbol z_{\ell-1}\right).
$$

</td>
<td style="width:50%; vertical-align:top;">

### General architecture of an FFNN
- **Input layer**: $\boldsymbol z_0 = \mathbf x = [x_1,\dots,x_n]^\top$
- **Output layer**: $\boldsymbol z_L = \hat{\mathbf y}$ (layer index $L$ is last)
- **Hidden layers**: $\ell=1,\dots,L-1$
- Each hidden layer $\ell$ has $K_\ell$ units
- $z_{k\ell}$ is unit $k$ in layer $\ell$; $\boldsymbol z_\ell$ stacks all units at layer $\ell$
- $b_{k\ell}$ is the bias for unit $k$ in layer $\ell$
- $\mathbf W_\ell$ is the weight matrix; $W_{jk\ell}$ is the weight from unit $j$ in layer $\ell-1$ to unit $k$ in layer $\ell$
- Hidden units are a **linear transform** of the previous layer followed by an **elementwise nonlinearity**
- $\varphi_\ell$ is the activation function (e.g., sigmoid, $\tanh$, ReLU, GELU)
- Typically **fully-connected** (every unit in $\ell-1$ connects to every unit in $\ell$)

</td>
</tr>
</table>

### Example
$\mathbf y = f(\mathbf x)$ with $\mathbf x\in\mathbb R^5,\ \mathbf y\in\mathbb R^1$,  
where $\varphi$ denotes the activation function.  
![](https://raw.githubusercontent.com/chaitragopalappa/MIE590-690D/main/images/supply_chain_nn.png)

## Optimizer

We can apply **stochastic gradient descent (SGD)**:
$$
\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t - \eta_t \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t).
$$
As above, $\boldsymbol{\theta}=\{ \mathbf b,\mathbf W_1,\dots,\mathbf W_L\}$, so SGD updates each parameter block.

**SGD in practice**
- If we can estimate $\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$ we can apply SGD.
- How to estimate gradients of a deep composition?
  - $\hat{\mathbf y} = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf x)$
  - $\mathcal{L}((\mathbf x, y), \boldsymbol{\theta}) = \|\hat{\mathbf y}-\mathbf y\|$
- Symbolic differentiation is tedious for such compositions.

### Automatic differentiation (autodiff)
- Autodiff breaks a function into a sequence of simple operators (a computational graph) and applies the **chain rule** to compute gradients efficiently.
- NN architectures naturally form computational graphs.
- Two modes: **forward** and **reverse**. For NNs, **reverse mode** is most efficient.
- Using reverse-mode autodiff with SGD is the **backpropagation** algorithm.

**Additional references**
- Baydin et al., *Automatic differentiation in machine learning: a survey*, JMLR 18 (2018) 1–43.
- Slides: https://dlsyscourse.org/slides/4-automatic-differentiation.pdf

### Autodiff for an FFNN

For an FFNN with a single hidden layer, a common squared-error objective is
$$
\mathcal{L}\big((\mathbf x, y), \boldsymbol{\theta}\big)
\;=\; \tfrac{1}{2}\,\big\|\,y - \mathbf W_2\, \varphi_2(\mathbf W_1 \mathbf x)\,\big\|_2^2.
$$

#### Feed-forward pass
We can view the computation as a composition of four functions:
$$
\mathcal{L} \;=\; f_4 \circ f_3 \circ f_2 \circ f_1, \qquad
\mathbf x_2 = f_1(\mathbf x,\theta_1)=\mathbf W_1 \mathbf x, \quad
\mathbf x_3 = f_2(\mathbf x_2,\varphi)=\varphi(\mathbf x_2), \quad
\mathbf x_4 = f_3(\mathbf x_3,\theta_3)=\mathbf W_2 \mathbf x_3, \quad
\mathcal{L} = f_4(\mathbf x_4,y)=\tfrac{1}{2}\|y-\mathbf x_4\|_2^2.
$$

*Figure embedded directly from the PML GitHub (Murphy):*  
![](https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.10.png)

#### Reverse-mode differentiation
$$
\frac{\partial \mathcal L}{\partial \theta_3}
= \frac{\partial \mathcal L}{\partial x_4}\,\frac{\partial x_4}{\partial \theta_3},\qquad
\frac{\partial \mathcal L}{\partial \theta_2}
= \frac{\partial \mathcal L}{\partial x_4}\,\frac{\partial x_4}{\partial x_3}\,\frac{\partial x_3}{\partial \theta_2},\qquad
\frac{\partial \mathcal L}{\partial \theta_1}
= \frac{\partial \mathcal L}{\partial x_4}\,\frac{\partial x_4}{\partial x_3}\,\frac{\partial x_3}{\partial x_2}\,\frac{\partial x_2}{\partial \theta_1}.
$$

### Backpropagation — general algorithm
```text
Algorithm: Backpropagation for an MLP with K layers (used inside an SGD loop)

# Forward pass
1:  x_1 := x
2:  for k = 1..K do
3:      x_{k+1} := f_k(x_k, θ_k)

# Backward pass
4:  u_{K+1} := 1
5:  for k = K..1 do
6:      g_k := u_{k+1}^T · ∂f_k(x_k, θ_k)/∂θ_k
7:      u_k^T := u_{k+1}^T · ∂f_k(x_k, θ_k)/∂x_k

# Output
8:  Return  L = x_{K+1},  ∇_x L = u_1,  and  { ∇_{θ_k} L = g_k : k = 1..K }.
```

### Backprop example for an FFNN with one hidden layer (sigmoid)
- Slides: https://github.com/chaitragopalappa/MIE590-690D/blob/main/images/BackProp_Example.pdf  
- Code: https://github.com/chaitragopalappa/MIE590-690D/blob/main/Codes/NN_Backpropagation_Vizual.ipynb

## Activation functions

| Name | Definition | Range | Reference |
|---|---|---|---|
| Sigmoid | $\sigma(a)=\dfrac{1}{1+e^{-a}}$ | $[0,1]$ |  |
| Hyperbolic tangent | $\tanh(a)=2\sigma(2a)-1$ | $[-1,1]$ |  |
| Softplus | $\operatorname{softplus}(a)=\log(1+e^a)$ | $[0,\infty)$ | GBB11 |
| ReLU | $\operatorname{ReLU}(a)=\max(a,0)$ | $[0,\infty)$ | GBB11; KSH12 |
| Leaky ReLU | $\max(a,0)+\alpha\,\min(a,0)$ | $(-\infty,\infty)$ | MHN13 |
| ELU | $\max(a,0)+\min\!\big(\alpha(e^{a}-1),0\big)$ | $(-\infty,\infty)$ | CUH16 |
| Swish | $a\,\sigma(a)$ | $(-\infty,\infty)$ | RZL17 |
| GELU | $a\,\Phi(a)$ | $(-\infty,\infty)$ | HG16 |

*Table Source:* Reproduced from Table 13.4 of **PML: An Introduction** (Murphy)

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.14_A.png" width="45%" />
  <img src="https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.14_B.png" width="45%" />
</p>

<p align="center"><em>Figure 13.14 from PML: An Introduction by Murphy (figures embedded from the textbook's GitHub repository)</em></p>

Additional reference: https://arxiv.org/abs/1811.03378

## Convergence
- Tune hyperparameters — learning rate, optimizer choice and its hyperparameters (see examples [here](https://colab.research.google.com/drive/1pPB_YTQ93pXyXctHPP-TMBN5woWJvV6J#scrollTo=yDiw8XyW3wh-)).
- Prefer non-saturating activations if you observe vanishing/exploding gradients.
- Try different initializations / random seeds.

## Regularization
*(Add your preferred regularizers: weight decay, dropout, early stopping, etc.)*

# MLP exercise problem for regression
```python
# TODO: Add exercise code or instructions here.
```

# MLP for classifying 2D data into 2 categories
TensorFlow Playground: https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.02424&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

Colab example (IMDB sentiment): https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/13/mlp_imdb_tf.ipynb

# MLP for heteroskedastic regression
*(Add notes/examples as needed.)*

# XAI
[SHAP tool](https://poloclub.github.io/webshap/?model=image)
