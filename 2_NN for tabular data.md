[copy](https://nbviewer.org/github/chaitragopalappa/MIE590-690D/blob/main/2_Neural_networks_for_tabular_data.ipynb)
<script type="text/javascript" async>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],  // <-- enable inline $...$ and \(..\)
    displayMath: [['$$','$$'], ['\\[','\\]']] // <-- display math
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
# Neural Networks for Structured Data or Tabular data

## Table of Contents
1. [Feed Forward Neural Network (FFNN) / Multi-layer Perceptrons (MLP)](#feed-forward-neural-network-ffnn--multi-layer-perceptrons-mlp)
   - [Outline](#outline)
2. [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
   - [Objective function](#objective-function)
   - [NN Model Architecture](#nn-model-architecture)
   - [Example](#example)
   - [Optimizer](#optimizer)
     - [Autodiff](#autodiff)
     - [Autodiff for FFNN](#autodiff-for-ffnn)
   - [Feed forward pass](#feed-forward-pass)
   - [Reverse mode differentiation](#reverse-mode-differentiation)
   - [Backpropagation - general algorithm](#backpropagation---general-algorithm)
   - [Using above example](#using-above-example)
   - [Backprop example for FFNN with one hidden layer using Sigmoid](#backprop-example-for-ffnn-with-one-hidden-layer-using-sigmoid)
   - [Activation functions](#activation-functions)
   - [Using Python NN packages](#using-python-nn-packages)
   - [Convergence](#convergence)
   - [Regularization](#regularization)
   - [MLP exercise problem for regression](#mlp-exercise-problem-for-regression)
     
## Feed Forward Neural Network (FFNN) / Multi-layer Perceptrons (MLP)
References:
* Chapter 13, Probabilistic Machine Learning: An Introduction by Kevin Murphy  

### Outline
* Objective function (minimize Loss)
* Model architecture (multiple layers- 'deep'; each node in each layer is a linear transformation of previous layer passed through an activation function)
* Optimizer (backprop that uses SGD and its variants in a Directed Acyclic Graph (DAG) structure)

# Multi-Layer Perceptron (MLP)
## **Objective function**
Suppose $$\mathbf{y}=f(\mathbf{{x}});\mathbf{x} \in \mathbb{R^n}; \mathbf{y} \in \mathbb{R^m}$$    
Objective function: $$Min\mathcal{L}(\mathbf{\theta}) =Min_\mathbf{\theta}||\mathbf{\hat{y}-y}||$$
* $$\mathbf{\hat{y}} $$ are predicted values from neural network
* $$\mathcal{L} $$ is the Loss   
* $\mathbf{\theta}=[\mathbf{b},\mathbf{W_1},\mathbf{W_2},...,\mathbf{W_L}]$ are neural network coefficients.

Thus, **'model fitting'** in context of NN refers to learning  $\mathbf{\theta}$ by minmizing the Loss function

## **NN Model Architecture**

<table>
<tr>
<td style="width:50%; vertical-align:top;">

### Model architecture:
$$z_L = f_L(z_{L-1}) = f_L(f_{L-1}(z_{L-2})) = f_L(f_{L-1}(\dots f_1(z_0)))$$  
$$\mathbf{x} \triangleq z_0, \quad \hat{y} \triangleq z_L$$  
or  
$$\hat{y} = f_L \circ f_{L-1} \circ f_{L-2} \dots \circ f_1$$  
where,  
$$\mathbf{z_l} = f_l(\mathbf{z_{l-1}}) = \varphi_l(\mathbf{b_l} + \mathbf{W_l} \mathbf{z_{l-1}})$$ (Vector form)
Rewriting in scalar form:  
$$z_{kl} = \varphi_l \Big( b_{kl} + \sum_{j=1}^{K_{l-1}} W_{jkl} z_{j(l-1)} \Big)$$  
The representation can be further simplified as:  
$$z_{kl} = \varphi_l \Big( \sum_{j=0}^{K_{l-1}} W_{jkl} z_{j(l-1)} \Big)$$  
by setting:  
$$z_{0(l-1)} = 1, \quad W_{0kl} = b_{kl}$$  
Putting back in vector form:  
$$\mathbf{z_l} = \varphi_l (\mathbf{W_l} \mathbf{z_{l-1}})$$  

</td>
<td style="width:50%; vertical-align:top;">

### General architecture of FFNN

* has an input layer representing the features with dimensionality equal to the number of input features,  
  i.e., $z_0 =\mathbf{x}= [x_1, \dots, x_n]$
* has an output layer representing the predicted variable with dimensionality equal to the output variable,  
  i.e., $z_L =\mathbf{\hat{y}}$; $L$ is the last layer
* has layers $l=1$ to $l=L-1$ as the hidden layers
* each hidden layer $l$ has $K_l$ number of nodes
* $z_{kl}$ is node $k$ in layer $l$
* $\mathbf{\mathit{z}}_l$ is a vector of nodes at layer $l$
* $b_{kl}$ is the bias node connecting to each node $k$ in layer $l$
* $\mathbf{W_l}$ is a matrix of coefficients; ${W_{jkl}}$ is coefficient from node $j$ in layer $l-1$ to node $k$ in layer $l$
* hidden units $z_l$ at each layer $l$ are a linear transformation of the hidden units at the previous layer  
  passed elementwise through an activation function
* $\varphi$ is an activation function (could be any differentiable function to add non-linearity; sigmoid, tanh, ReLU are common)
* typically a **fully connected** feedforward network (arrow from every node in layer $l-1$ to every node in layer $l$)

</td>
</tr>
</table>

### Example
 $\mathbf{y}=f(\mathbf{x});\mathbf{x}\in \mathbb{R}^5; \mathbf{y}\in\mathbb{R}^1$  where $\sigma=  φ_l (b_{kl} +\sum_{j=1}^{K_{l-1}}{W}_{jkl}z_{jl−1}) $ denoting the activiation function
 ![](https://raw.githubusercontent.com/chaitragopalappa/MIE590-690D/main/images/supply_chain_nn.png)

 ## **Optimizer**
We can apply [stochastic gradient descent (SGD)](https://colab.research.google.com/drive/1pPB_YTQ93pXyXctHPP-TMBN5woWJvV6J#scrollTo=90ILVxtj9MRl)   
$\mathbf{\theta_{t+1}} = \mathbf{\theta_t} − η_t∇_\mathbf{\theta}\mathcal{L}(\mathbf{\theta_t})$  
As discussed above
* $\mathbf{\theta}=[\mathbf{b},\mathbf{W_1},\mathbf{W_2},...,\mathbf{W_L}]$ are the coefficients of the neural network, so we will have to apply SDG to each coefficient


SGD application:
* If we can estimate $∇_\mathbf{\theta}\mathcal{L}(\mathbf{\theta}_t)$ we can apply SDG
* How to estimate gradients of such complex function?
  * $\hat{y}= f_L \circ f_{L-1}\circ f_{L-2}.....\circ f_1$
  * $\mathcal{L}= ||{\hat{y}-y}||$ also written as $\mathcal{L((\mathbf{x}, y),\theta)}= ||{\hat{y}-y}||$
  * $\mathcal{L} \equiv f_L \circ f_{L-1}\circ f_{L-2}.....\circ f_1$
* Symbolic differentiation can be tedious for deeper netwroks given the complex function (composition of functions)

### Autodiff
 * Autodiff or automatic differentiation breaks a function into sequence of simple operators (that can be represented in a computational graph structure) and applies chain rule from calculus to sequentially calcuate the gradient of each operator
  * The architecture of the neural network nuturally has a computational graph structure.
  *  The sequence of derivative calculations can be forward mode or reverse mode.
  * Given the nature of the NN architecture reverse model autodiff is more applicable here

  ### Additional reference:
  * AutoDiff: Baydan et. al., Journal of Machine Learning Research 18 (2018) 1-43
  * [Slides](https://dlsyscourse.org/slides/4-automatic-differentiation.pdf)

*The above full approach of using reverse mode autodiff with SGD is called BackProp (backpropogation) algorithm*

### **Autodiff for FFNN**

Example: For a FFNN with a single hidden-layer the objective function is
 $\mathcal{L((\textbf{x},y),\theta)}= ||{\hat{y}-y}||= \frac{1}{2}||y-\mathbf{W}_2  φ_2 (\mathbf{W}_1\mathbf{x})||_2^2 $


 ### Feed forward pass of the NN calculates the value of $\mathcal{L}$
 The objective function can be rewritten into sequence of 4 operators (4 layers from perspective of autodiff computational graph - do not confuse with NN architecture layers)

*Figure embeded directly from [GitHub PML by Murphy](https://github.com/probml/pml-book/blob/main/book1-figures/Figure_13.10.png)*
 ![](https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.10.png)



### Feed forward pass

 $\mathcal{L}=f_4 \circ f_3 \circ f_2 \circ f_1$    
 $(\mathbf{x}=\mathbf{x}_1)$  
 $\mathbf{x}_2= f_1(\mathbf{x},\theta_1 )=\mathbf{W}_1\mathbf{x} $    
 $\mathbf{x}_3= f_2(\mathbf{x_2},{φ})={φ}(\mathbf{x_2})$  
 $\mathbf{x}_4= f_3(\mathbf{x_3},\theta_3 )=\mathbf{W}_2\mathbf{x}_3$  
 $o=\mathcal{L} = f_4 (\mathbf{x_4},y)=\frac{1}{2}||y-x_4||_2^2 $

### Reverse mode differentiation
$\frac{\partial \mathcal{L}}{\partial \theta_3} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial \theta_3}$  
$\frac{\partial \mathcal{L}}{\partial \theta_2}  = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial \theta_2}$  
$\frac{\partial \mathcal{L}}{\partial \theta_1}
= \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial x_2} \frac{\partial x_2}{\partial \theta_1}
$

### **Backpropagation -general algorithm**
$$
\begin{array}{l}
\textbf{Algorithm: Backpropagation for an MLP with $K$ layers used inside SDG loop}\\     
\text{// Forward pass} \\
1: \quad x_1 := x \\
2: \quad \text{for } k = 1 : K \ \text{do} \\
3: \quad\quad x_{k+1} = f_k(x_k, \theta_k) \\  
\text{// Backward pass} \\
4: \quad u_{K+1} := 1 \\
5: \quad \text{for } k = K : 1 \ \text{do} \\
6: \quad\quad g_k := u_{k+1}^\top \frac{\partial f_k(x_k, \theta_k)}{\partial \theta_k} \\
7: \quad\quad u_k^\top := u_{k+1}^\top \frac{\partial f_k(x_k, \theta_k)}{\partial x_k} \\ 
\text{// Output} \\
8: Return \quad \mathcal{L} = x_{K+1}, \quad \nabla_x \mathcal{L} = u_1, \quad \{\nabla_{\theta_k} \mathcal{L} = g_k : k = 1 : K\}
\end{array}
$$

### Using above example
$$
\begin{array}{l}
\textbf{Algorithm: Backpropagation for an MLP with $K$ layers used inside SDG loop} \\
\text{// Forward pass} \\
\mathbf{x_1} := \mathbf{x} \\
\mathbf{x_2}= f_1(\mathbf{x},\theta_1 )=\mathbf{W_1}\mathbf{x}    \\
\mathbf{x_3}= f_2(\mathbf{x_2},{φ})={φ}(\mathbf{x_2})\\
\mathbf{x_4}= f_3(\mathbf{x_3},\theta_3 )=\mathbf{W_2}\mathbf{x_3} \\
o=\mathcal{L} = f_4 (\mathbf{x_4},y)=\frac{1}{2}||y-x_4||_2^2 \\
\text{// Backward pass} \\
\end{array}
$$

$$u_{4+1} := 1$$
$$\frac{\partial \mathcal{L}}{\partial \theta_4} = g_4 := {u_{4+1}^T} \frac{\partial f_4(x_4, \theta_4)}{\partial \theta_4} = {u_{4+1}^T} \frac{\partial \mathcal{L}}{\partial \theta_4} = 0 $$
$$\frac{\partial \mathcal{L}}{\partial x_4} = {u_4^T} := {u_{4+1}^T} \frac{\partial f_4(x_4, \theta_4)}{\partial x_4} = {u_{4+1}^T} \frac{\partial \mathcal{L}}{\partial x_4} = -(y - x_4) $$
$$\frac{\partial \mathcal{L}}{\partial \theta_3} = g_3 := {u_{3+1}^T} \frac{\partial f_3(x_3, \theta_3)}{\partial \theta_3} = {u_{3+1}^T} \frac{\partial x_4}{\partial \theta_3} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial \theta_3} = \frac{\partial \mathcal{L}}{\partial x_4} x_3 \; (\equiv \frac{\partial \mathcal{L}}{\partial W_2}) $$
$$\frac{\partial \mathcal{L}}{\partial x_3} = {u_3^T} := {u_{3+1}^T} \frac{\partial f_3(x_3, \theta_3)}{\partial x_3} = {u_{3+1}^T} \frac{\partial x_4}{\partial x_3} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} = \frac{\partial \mathcal{L}}{\partial x_4} W_2 $$
$$\frac{\partial \mathcal{L}}{\partial \theta_2} = g_2 := {u_{2+1}^T} \frac{\partial f_2(x_2, \theta_2)}{\partial \theta_2} = {u_{2+1}^T} \frac{\partial x_3}{\partial \theta_2} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial \theta_2} = \frac{\partial \mathcal{L}}{\partial x_3} \frac{\partial \varphi}{\partial \varphi} 
$$
$$
\frac{\partial \mathcal{L}}{\partial x_2} = {u_2^T} := {u_{2+1}^T} \frac{\partial f_2(x_2, \theta_2)}{\partial x_2} = {u_{2+1}^T} \frac{\partial x_3}{\partial x_2} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial x_2} = \frac{\partial \mathcal{L}}{\partial x_3} \frac{\partial \varphi}{\partial x_2} $$
$$\frac{\partial \mathcal{L}}{\partial \theta_1} = g_1 := {u_{1+1}^T} \frac{\partial f_1(x_1, \theta_1)}{\partial \theta_1} = {u_{1+1}^T} \frac{\partial x_2}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial x_2} \frac{\partial x_2}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial x_2} x_1 \; (\equiv \frac{\partial \mathcal{L}}{\partial W_1}) $$
$$\frac{\partial \mathcal{L}}{\partial x_1} = {u_1^T} := {u_{1+1}^T} \frac{\partial f_1(x_1, \theta_1)}{\partial x_1} = {u_{1+1}^T} \frac{\partial x_2}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial x_2} \frac{\partial x_2}{\partial x_1}
$$

$$
\begin{array}{l}\\
\text{// Output} \\
Return \quad \mathcal{L} = x_{K+1}, \quad \nabla_x \mathcal{L} = u_1= \frac{\partial \mathcal{L}}{\partial x_1}, \quad \{\nabla_{\theta_k} \mathcal{L} = g_k = \frac{\partial \mathcal{L}}{\partial \theta_k}: k = 1 : K\} \\
\end{array}
$$

### Backprop example for FFNN with one hidden layer using Sigmoid activation
[Slides](https://github.com/chaitragopalappa/MIE590-690D/blob/main/suppl_files/2a_BackProp_Example.pdf)  
[Code-Using Symbolic differentiation](https://github.com/chaitragopalappa/MIE590-690D/blob/main/Codes/2b_NN_Backpropagation_Vizual.ipynb)  
**[Code-Using AutoDiff]**

## **Activation functions**
Common used functions
| Name | Definition | Range | Reference |
|------|------------|-------|-----------|
| Sigmoid | $\sigma(a) = \frac{1}{1 + e^{-a}}$ | [0, 1] |  |
| Hyperbolic tangent | $\tanh(a) = 2\sigma(2a) - 1$ | [-1, 1] |  |
| Softplus | $\sigma_{+}(a) = \log(1 + e^{a})$ | [0, $\infty$) | [GBB11] |
| Rectified linear unit | $\mathrm{ReLU}(a) = \max(a, 0)$ | [0, $\infty$) | [GBB11; KSH12] |
| Leaky ReLU | $\max(a, 0) + \alpha \min(a, 0)$ | $(-\infty, \infty)$ | [MHN13] |
| Exponential linear unit | $\max(a, 0) + \min\big(\alpha(e^{a} - 1), 0\big)$ | $(-\infty, \infty)$ | [CUH16] |
| Swish | $a \, \sigma(a)$ | $(-\infty, \infty)$ | [RZL17] |
| GELU | $a \, \Phi(a)$ | $(-\infty, \infty)$ | [HG16] |

*Table Source:* Reproduced from Table 13.4 PML: An Introduction by Murphy

<p align="center">
  <img src="https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.14_A.png" width="45%" />
  <img src="https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.14_B.png" width="45%" />
</p>

<p align="center"><em>Figure 13.14 from PML: An Introduction by Murphy (Directly embedded figure from the textbook's GitHub repository)</em></p>
Addtional Reference: https://arxiv.org/abs/1811.03378

## **Using Python NN packages**
Using PyTorch   
[2b_NN_PyTorch_Basic.ipynb]()

## Convergence
* Tune hyperparmeters -learning rate, different optimizers and their hyperparamters (See optimizers and convergence properties [1a_Mathematical Foundations of ML](https://colab.research.google.com/drive/1pPB_YTQ93pXyXctHPP-TMBN5woWJvV6J#scrollTo=yDiw8XyW3wh-)
* Pick non-saturating activation functions if issues of vanishing/exploding gradient
* Initialize with different initial weights (or use different random seeds if fixing the seed)

## Regularization

# MLP exercise problem for regression





