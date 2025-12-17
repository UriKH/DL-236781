r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

A. X has shape (64, 1024) and Y has shape (64, 512), The Jacobian tensor $\pderiv{\mat{Y}}{\mat{X}}$ describes, for each output element, how it changes with respect to each input element:

$(\pderiv{\mat{Y}}{\mat{X}})_{n,o,m,i} = \pderiv{\mat{Y}_{n,o}}{\mat{X}_{m,i}}$

Therefore, the Jacobian tensor will have shape (64, 512, 64, 1024).

B. If we view the Jacobian $\pderiv{\mat{Y}}{\mat{X}}$ as a 2D block matrix, we will get $N \times N$ blocks of (out_features × in_features) such that the block (i,j) is $\pderiv{\mat{Y}_i}{\mat{X}_j}$. For a linear layer $Y=XW^T$, each output $\mat{Y}_k$ depends only on its own input sample $\mat{X}_k$ we get:
\begin{cases}
    \pderiv{\mat{Y}_i}{\mat{X}_j}=W & i = j\\
    \pderiv{\mat{Y}_i}{\mat{X}_j}=0 & i \ne j
\end{cases}
The structure is a block - diagonal matrix with N diagonal blocks, equal to $W$ and all other entries are zero.

C. Yes, apart from VJP we can use the Jacobian’s block structure. As we say previously, $\pderiv{\mat{Y}_i}{\mat{X}_j}$ is a block - diagonal matrix with N diagonal blocks, equal to $W$ and all other entries are zero. Therefore, there is no need to materialize the full Jacobian of shape, we can just represent it by storing only $W$.

D. Given the gradient of the output w.r.t. some downstream scalar loss $L$, $\delta\mat{Y} := \pderiv{L}{\mat{Y}}$ we can use the chain rule to calculate the downstream gradient w.r.t. the input ($\delta\mat{X}$) without materializing the Jacobian:

$\delta \mat{X}_{n,i} = \pderiv{L}{\mat{X}_{n,i}} = \sum_{o=1}^{O} \pderiv{L}{\mat{Y}_{n,o}} \, \pderiv{\mat{Y}_{n,o}}{\mat{X}_{n,i}}$

$\mat{Y}_{n,o} = \sum_{j=1}^J \mat{X}_{n,j}\mat{W}_{j,o}^T$

Therefore, 

$\delta \mat{X}_{n,i} = \sum_{o=1}^{O} \delta \mat{Y}_{n,o}\mat{W}_{j,o}^T = \sum_{o=1}^{O} \delta \mat{Y}_{n,o}\mat{W}_{o,j}$

In matrix form for the whole batch:

$\delta \mat{X} = \mat{Y}\mat{W}$

E. W has shape (512, 1024) and Y has shape (64, 512), The Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$ describes, for each output element, how it changes with respect to each weight element:

$(\pderiv{\mat{Y}}{\mat{W}})_{n,o,p,i} = \pderiv{\mat{Y}_{n,o}}{\mat{W}_{p,i}}$

Therefore, the Jacobian tensor will have shape (64, 512, 512, 1024). If we were to make it into a block matrix, the shape of the block will be 

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr = 0.01
    reg = 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.0035
    lr_vanilla = 0.04
    lr_momentum = 0.005
    lr_rmsprop = 0.0002
    reg = 0.002
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.005
    lr = 0.0005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible for the test loss to decrease while the test accuracy also decreases. Cross-entropy loss depends on the predicted probabilities, while accuracy only depends on whether the argmax prediction is correct. For example, suppose we have 2 test samples $\{ x_1, x_2 \}$, both with true label 1.

$$\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - {\vectr{y}} \log(\hat{\vec{y}})$$

Epoch 1 – 100% accuracy:

$p_1 = (0.49, 0.51)$

$p_2 = (0.49, 0.51)$

$\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - (0,1)^T \cdot log(0.51, 0.51) = -log(0.51) = 0.673$

Epoch 2 – 50% accuracy:

$p_1 = (0.01, 0.99)$

$p_2 = (0.51, 0.49)$

$\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - (0,1)^T \cdot log(0.49, 0.99) = -log(0.99) = 0.01$
"""

part2_q3 = r"""
**Your answer:**
1. Similarities of gradient descent (GD) and stochastic gradient descent (SGD):

(1) Both aim to minimize the same loss function

(2) Same update rule $\theta \leftarrow \theta - \eta \cdot gradient$

Differences of gradient descent (GD) and stochastic gradient descent (SGD):

(1) How they compute the gradient - GD uses all data while SGD pick one sample from a uniform distrubution on tha dataset.

(2) Cost - GD goes through all N examples to compute one gradient while SGD computes the gradient w.r.t only one sample, so each step is much faster.

(3) Noise - GD get deterministic path and the loss usually decreases smoothly while in SGD each gradient is noisy (since it is computed on different samples each epoch) and the randomness in the sampling gives different paths.

2. 

3.

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 100
    activation = 'relu'
    out_activation = 'sigmoid'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.1
    weight_decay = 0.003
    momentum = 0.003
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.1
    loss_fn = torch.nn.CrossEntropyLoss()
    weight_decay = 0.001
    momentum = 0.02
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""