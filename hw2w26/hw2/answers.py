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

$(\pderiv{\mat{Y}}{\mat{X}}){n,o,m,i} = \pderiv{\mat{Y}{n,o}}{\mat{X}_{m,i}}$

Therefore, the Jacobian tensor will have shape (64, 512, 64, 1024).

B. If we view the Jacobian $\pderiv{\mat{Y}}{\mat{X}}$ as a 2D block matrix, we will get $N \times N$ blocks of (out_features × in_features) such that the block (i,j) is $\pderiv{\mat{Y}_i}{\mat{X}_j}$. For a linear layer $Y=XW^T$, each output $\mat{Y}_k$ depends only on its own input sample $\mat{X}_k$ we get:
\begin{cases}
    \pderiv{\mat{Y}_i}{\mat{X}_j}=W & i = j\\
    \pderiv{\mat{Y}_i}{\mat{X}_j}=0 & i \ne j
\end{cases}
The structure is a block - diagonal matrix with N diagonal blocks, equal to $W$ and all other entries are zero.

C. Yes, apart from VJP we can use the Jacobian’s block structure. As we say previously, $\pderiv{\mat{Y}_i}{\mat{X}_j}$ is a block - diagonal matrix with N diagonal blocks, equal to $W$ and all other entries are zero. Therefore, there is no need to materialize the full Jacobian of shape, we can just represent it by storing only $W$.

D. Given the gradient of the output w.r.t. some downstream scalar loss $L$, $\delta\mat{Y} := \pderiv{L}{\mat{Y}}$ we can use the chain rule to calculate the downstream gradient w.r.t. the input ($\delta\mat{X}$) without materializing the Jacobian:

$\delta \mat{X}{n,i} = \pderiv{L}{\mat{X}{n,i}} = \sum_{o=1}^{O} \pderiv{L}{\mat{Y}{n,o}} \, \pderiv{\mat{Y}{n,o}}{\mat{X}_{n,i}}$

$\mat{Y}{n,o} = \sum{j=1}^J \mat{X}{n,j}\mat{W}{j,o}^T$

Therefore, 

$\delta \mat{X}{n,i} = \sum{o=1}^{O} \delta \mat{Y}{n,o}\mat{W}{j,o}^T = \sum_{o=1}^{O} \delta \mat{Y}{n,o}\mat{W}{o,j}$

In matrix form for the whole batch:

$\delta \mat{X} = \mat{Y}\mat{W}$

E. W has shape (512, 1024) and Y has shape (64, 512), The Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$ describes, for each output element, how it changes with respect to each weight element:

$(\pderiv{\mat{Y}}{\mat{W}}){n,o,p,i} = \pderiv{\mat{Y}{n,o}}{\mat{W}_{p,i}}$

Therefore, the Jacobian tensor will have shape (64, 512, 512, 1024). If we were to make it into a block matrix, each block has shape (64, 1024).




If we were to make it into a block matrix, each block contains all derivatives across the remaining indices $(n,i)$:

$\text{block}(o,p) = \left[\pderiv{Y_{n,o}}{W_{p,i}}\right]{n=1}^{64}{}{i=1}^{1024}.$

Therefore, each block has shape $(64,1024)$.

"""

part1_q2 = r"""
**Your answer:**
Yes, second order derivatives (Hessian) can be helpful for optimization (but are not commonly used as it is quite expensive to compute unlike the gradient).

When second order information is useful when the condition number is high.
If the condition number is high this means that in some directions the optimization landscape is much more steep. 

This means we would like to take very big learning rate in order to overcome the low steepness in one direction but a very small learning rate to overcome the high steepness in the other direction. In order to solve this contradiction, we can use Newton's method which utilizes the Hessian matrix in order to "normalize" gradient step in this case.
"""



# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
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
    wstd = 0.12
    lr_vanilla = 0.0155
    lr_momentum = 0.0019 # 0.0025 # 0.0015
    lr_rmsprop = 0.0001255 #0.000125 #0.0001
    reg = 0.024
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
    wstd = 0.05
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Yes, the graphs match the theoretical expectations.

- Without Dropout (blue curve): In the train_acc graph, the accuracy reaches fast to nearly 80% However, in the test_acc graph, the performance is at a much lower level (~26%). In addition, the test_loss graph shows the loss decreasing initially but then starting to rise again. This configurations demonstrate overfitting - the model is failing to generalize to new data.

- With Dropout (orange and green curves): The dropout successfully acts as regularization as we can see that the gap between training and testing performance is significantly smaller.

2. Comparing dropout=0.4 to dropout=0.8:

- Low Dropout (0.4): It applies enough regularization to prevent the overfitting, but it preserves enough information to allow the model to learn effectively. It achieves the highest test accuracy and appears to be optimal.

- High Dropout (0.8): This setting is too aggressive and leading to underfitting. In the train_acc graph, the green line struggles to exceed 35% accuracy. Because the model drops 80% of its connections during each pass, it lacks the capacity to learn complex features from the training data. 
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

- Both aim to minimize the same loss function

- Same update rule $\theta \leftarrow \theta - \eta \cdot gradient$

Differences of gradient descent (GD) and stochastic gradient descent (SGD):

- How they compute the gradient - GD uses all data while SGD pick one sample from a uniform distrubution on tha dataset.

- Cost - GD goes through all N examples to compute one gradient while SGD computes the gradient w.r.t only one sample, so each step is much faster.

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
    out_activation = 'none'
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
    lr = 0.05
    weight_decay = 0.01
    momentum = 0.001
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
1. The model preformed quite bad on both images.
In the first image it wronglly classified one of the dolphins as a surfboard while the other two as people, and in the second image classified two of the dogs as cats while it didn't find the cat in the image at all.

The confidances of the model on the first image where as follows:
- 0.47 person (true = dolphin)
- 0.9 person (true = dolphin)
- 0.37 surfboard (true = dolphin)

In the second image:
- 0.65 cat (true = dog)
- 0.39 cat (true = dog)
- 0.5 dog (true = dog)

2. **Reasons for failure:**
- The model hasn't been trained on dolphin images and thus is not able to classify images as dolphins at all.
- The model hasn't been trained on special illumination conditions of dolphins and thus unable to detect the dolphins using the dolphin features it was able to detect on training.
- The model hasn't been trained on different breeds of dogs therefore unable to classify two of the dogs. Probably the same is true for cats as it mistook the dogs for cats with quite high confidance for one of them while it didn't recognize the true cat in the image at all.

**Possible solutions:**
- Train the model using more data and diverse data (solving dog detection + cat detection).
- Train the model on augmented data with dicersity of augmentations like bad illumination of the object / blurry images of it etc.

3. In order to attack an object detection model such as YOLO, we must first define the attack objective, for example: object disappearance, misclassification, bounding box distortion, or generating false positives.

PGD aims to find a small adversarial perturbation added to the input image that significantly degrades the model’s predictions while keeping the perturbation bounded (under some constraints).

PGD is a white-box attack, meaning the attacker has full access to the model architecture and trained weights and thus can compute gradients with respect to the input. This allows defining or reusing a suitable detection loss that aligns with the attack objective.

The attack is performed by iteratively updating the input image in the direction of the gradient that maximizes the attack loss, while projecting the perturbed image back to the allowed bounded region. This process is repeated until the attack succeeds.
"""


part6_q2 = r""" IGNORE THIS """


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