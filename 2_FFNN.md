# Chapter 13: Neural Networks for Structured Data or Tabular data
(Feed Forward Neural Network)

### Outline
* Objective function (minimize Loss)
* Model architecture (multiple layers- 'deep'; each node in each layer is a linear transformation of previous layer passed through an activation function)
* Optimizer (backprop that uses SGD and its variants in a Directed Acyclic Graph (DAG) structure)

# Multi-Layer Perceptron (MLP)

### **Obejctive function**
Suppose ${\mathbf{y}}=f(\mathbf{{x}});\mathbf{x}\in \mathbb{R}^n; \mathbf{y}\in\mathbb{R}^m$\
Objective function: $ Min\mathcal{L}(\mathbf{\theta}) =Min_\mathbf{\theta}||\mathbf{\hat{y}-y}||$
* $\mathbf{\hat{y}} $ are predicted values from neural network
* $\mathcal{L} $ is the Loss   
* $\mathbf{\theta}=[\mathbf{b},\mathbf{W_1},\mathbf{W_2},...,\mathbf{W_L}]$ are neural network coefficients.

## **NN Model Architecture**
### Model architecture:
$\hat{y}  \triangleq  z_L =f_L(z_{L−1})= f_L(f_{L-1}(z_{L−2}))=f_L(f_{L-1}(.....f_1(z_{0}))) ; z_0 \triangleq \mathbf{x}$\
or\
$\hat{y}= f_L \circ f_{L-1}\circ f_{L-2}.....\circ f_1$\
where,\
$\mathbf{\mathit{z}}*l = f_l(\mathbf{\mathit{z}}*{l−1}) = φ_l(\mathbf{\mathit{b}}_l +\mathbf{W}*lz*{l−1})$  (Vector form)  

> Rewriting in scalar form\
$z_{kl} = φ_l (b_{kl} +\sum_{j=1}^{K_{l-1}}{W}*{jkl}z*{jl−1}) $\
The representation can be further simplified as\
$z_{kl} = φ_l (\sum_{j=0}^{K_{l-1}}{W}*{jkl}z*{jl−1}) $  by setting $z_{0l−1}=1$ and ${W}*{0kl}=b*{kl}$  

Putting back in vector form:\
$\mathbf{\mathit{z}}_l = φ_l (\mathbf{W}*l z*{l−1}) $

General architecture of FFNN:
* has an input layer representing the features with dimensionality equal to the number of input features, i.e., $z_0 =\mathbf{x}= [x_1, ...x_n]$;
* has an output layer representing the predicted variable with dimentionaly equal to the output variable, i.e., $z_L =\mathbf{\hat{y}}$; $L$ is the last layer
* has layers $l=1$ to $l=L-1$ as the hidden layers
* each hidden layer $l$ has $K_l$ number of nodes
* $z_{kl}$ is node $k$ in layer $l$
* $\mathbf{\mathit{z}}_l$ is a vector of nodes at layer $l$
* $b_{kl}$ is the bias node connecting to each node $k$ in layer $l$
* $\mathbf{W}*l$ is a matrix of coefficients; ${W}*{jkl}$ is coefficent from node $j$ in layer $l-1$ to node $k$ in layer $l$
* hidden units $z_l$ at each layer $l$ is a linear transformation of the hidden units at the previous layer passed elementwise through an activation function.
*$\varphi$ is an activation function (could be any differntiable function to add non-linearity to the model; sigmoid, tanh, ReLU are commonly used)
* it is typically a **fully connected** feed forward network (arrow from every node in layer $l-1$ to every node in layer $l$)

|  |
| --- |

## **NN Model Architecture**

### Model architecture:
$z_L =f_L(z_{L−1})= f_L(f_{L-1}(z_{L−2}))=f_L(f_{L-1}(\dots f_1(z_{0}))) $  

> $ \mathbf{x} \triangleq z_0  $\
$\hat{y}  \triangleq  z_L$

or\
$\hat{y}= f_L \circ f_{L-1}\circ f_{L-2} \dots \circ f_1$  

where,\
$\mathbf{\mathit{z}}*l = f_l(\mathbf{\mathit{z}}*{l−1}) = \varphi_l(\mathbf{\mathit{b}}_l +\mathbf{W}*l z*{l−1})$  (Vector form)  

> Rewriting in scalar form\
$z_{kl} = \varphi_l (b_{kl} +\sum_{j=1}^{K_{l-1}}{W}*{jkl}z*{jl−1})$\
The representation can be further simplified as\
$z_{kl} = \varphi_l (\sum_{j=0}^{K_{l-1}}{W}*{jkl}z*{jl−1})$\
by setting $z_{0l−1}=1$ and ${W}*{0kl}=b*{kl}$  

Putting back in vector form:\
$\mathbf{\mathit{z}}_l = \varphi_l (\mathbf{W}*l z*{l−1})$

|

### General architecture of FFNN

* has an input layer representing the features with dimensionality equal to the number of input features,\
i.e., $z_0 =\mathbf{x}= [x_1, \dots, x_n]$
* has an output layer representing the predicted variable with dimensionality equal to the output variable,\
i.e., $z_L =\mathbf{\hat{y}}$; $L$ is the last layer
* has layers $l=1$ to $l=L-1$ as the hidden layers
* each hidden layer $l$ has $K_l$ number of nodes
* $z_{kl}$ is node $k$ in layer $l$
* $\mathbf{\mathit{z}}_l$ is a vector of nodes at layer $l$
* $b_{kl}$ is the bias node connecting to each node $k$ in layer $l$
* $\mathbf{W}*l$ is a matrix of coefficients; ${W}*{jkl}$ is coefficient from node $j$ in layer $l-1$ to node $k$ in layer $l$
* hidden units $z_l$ at each layer $l$ are a linear transformation of the hidden units at the previous layer\
passed elementwise through an activation function
* $\varphi$ is an activation function (could be any differentiable function to add non-linearity; sigmoid, tanh, ReLU are common)
* typically a **fully connected** feedforward network (arrow from every node in layer $l-1$ to every node in layer $l$)

### Example
$\mathbf{y}=f(\mathbf{x});\mathbf{x}\in \mathbb{R}^5; \mathbf{y}\in\mathbb{R}^1$  where $\sigma=  φ_l (b_{kl} +\sum_{j=1}^{K_{l-1}}{W}*{jkl}z*{jl−1}) $ denoting the activiation function
![](https://raw.githubusercontent.com/chaitragopalappa/MIE590-690D/main/images/supply_chain_nn.png)

## **Optimizer**
We can apply [stochastic gradient descent (SGD)](https://colab.research.google.com/drive/1pPB_YTQ93pXyXctHPP-TMBN5woWJvV6J#scrollTo=90ILVxtj9MRl)\
$\mathbf{\theta}_{t+1} = \mathbf{\theta}*t − η_t∇*\mathbf{\theta}\mathcal{L}(\mathbf{\theta}_t)$\
As discussed above
* $\mathbf{\theta}=[\mathbf{b},\mathbf{W_1},\mathbf{W_2},...,\mathbf{W_L}]$ are the coefficients of the neural network, so we will have to apply SDG to each coefficient

SGD application:
* If we can estimate $∇_\mathbf{\theta}\mathcal{L}(\mathbf{\theta}_t)$ we can apply SDG
* How to estimate gradients of such complex function?
  * $\hat{y}= f_L \circ f_{L-1}\circ f_{L-2}.....\circ f_1$
  * $\mathcal{L}= ||{\hat{y}-y}||$ also written as $\mathcal{L((\mathbf{x},y),\theta)}= ||{\hat{y}-y}||$
  * $\mathcal{L} \equiv f_L \circ f_{L-1}\circ f_{L-2}.....\circ f_1$
* Symbolic differentiation is tedious given the complex function (composition of functions)

### Autodiff
* Autodiff or automatic differentiation breaks a function into sequence of simple operators (that can be represented in a computational graph structure) and applies chain rule from calculus to sequentially calcuate the gradient of each operator
* The architecture of the neural network nuturally has a computational graph structure.
* The sequence of derivative calculations can be forward mode or reverse mode.
* Given the nature of the NN architecture reverse model autodiff is more applicable here

*The above full approach of using reverse mode autodiff with SGD is called BackProp (backpropogation) algorithm*

## **Autodiff for FFNN**

Example: For a FFNN with a single hidden-layer the objective function is
$\mathcal{L((\textbf{x},y),\theta)}= ||{\hat{y}-y}||= \frac{1}{2}||y-\mathbf{W}_2  φ_2 (\mathbf{W}_l\mathbf{x})||_2^2 $

### Feed forward pass of the NN calculates the value of $\mathcal{L}$
The objective function can be rewritten into sequence of 4 operators (4 layers from perspective of autodiff computational graph - do not confuse with NN architecture layers)

*Figure embeded directly from [GitHub PML by Murphy](https://github.com/probml/pml-book/blob/main/book1-figures/Figure_13.10.png)*
![](https://raw.githubusercontent.com/probml/pml-book/main/book1-figures/Figure_13.10.png)

### Feed forward pass

$\mathcal{L}=f_4 \circ f_3 \circ f_2 \circ f_1$\
$\mathbf{x}_2= f_1(\mathbf{x},\theta_1 )=\mathbf{W}_1\mathbf{x}$\
$\mathbf{x}_3= f_2(\mathbf{x_2},{φ})={φ}(\mathbf{x_2})$\
$\mathbf{x}_4= f_3(\mathbf{x_3},\theta_3 )=\mathbf{W}_2\mathbf{x}_3$\
$\mathcal{L} = f_4 (\mathbf{x_4},y)=\frac{1}{2}||y-x_4||_2^2 $

### Reverse mode differentiation
> $\frac{\partial \mathcal{L}}{\partial \theta_3} = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial \theta_3}$\
$ \frac{\partial \mathcal{L}}{\partial \theta_2}  = \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial \theta_2}$\
$\frac{\partial \mathcal{L}}{\partial \theta_1}
= \frac{\partial \mathcal{L}}{\partial x_4} \frac{\partial x_4}{\partial x_3} \frac{\partial x_3}{\partial x_2} \frac{\partial x_2}{\partial \theta_1}
$

## **Backpropagation -general algorithm**
$$
\begin{array}{l}
\textbf{Algorithm: Backpropagation for an MLP with $K$ layers used inside SDG loop} \\[2mm]
\text{// Forward pass} \\
1: \quad x_1 := x \\
2: \quad \text{for } k = 1 : K \ \text{do} \\
3: \quad\quad x_{k+1} = f_k(x_k, \theta_k) \\[2mm]
\text{// Backward pass} \\
4: \quad u_{K+1} := 1 \\
5: \quad \text{for } k = K : 1 \ \text{do} \\
6: \quad\quad g_k := u_{k+1}^\top \frac{\partial f_k(x_k, \theta_k)}{\partial \theta_k} \\
7: \quad\quad u_k^\top := u_{k+1}^\top \frac{\partial f_k(x_k, \theta_k)}{\partial x_k} \\[2mm]
\text{// Output} \\
8: Return \quad \mathcal{L} = x_{K+1}, \quad \nabla_x \mathcal{L} = u_1, \quad \{\nabla_{\theta_k} \mathcal{L} = g_k : k = 1 : K\}
\end{array}
$$

### Backprop example for FNNN with one hidden layer using Sigmoid activation
[Slides](https://github.com/chaitragopalappa/MIE590-690D/blob/main/images/BackProp_Example.pdf)\
[Code](https://github.com/chaitragopalappa/MIE590-690D/blob/main/Codes/NN_Backpropagation_Vizual.ipynb)

# MLP for classifying 2D data into 2 categories

[Tensor playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.02424&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
