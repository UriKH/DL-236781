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

2. Yes, momentum can be used with GD. Momentum can accelerate convergence by accumulating velocity in directions with consistent gradients often leading to faster and stable convergence. The main difference from SGD is that in GD momentum is not primarily “noise smoothing”.

3.  A) Let the dataset be $\mathcal{D}=\{x_i\}_{i=1}^N$ and the loss be $\ell(\theta; x_i)$.

$L(\theta)=\sum_{i=1}^N \ell(\theta;x_i)$

hence its gradient is:

$\nabla_\theta L(\theta)=\sum_{i=1}^N \nabla_\theta \ell(\theta;x_i)$

Assume we partition the dataset into disjoint batches $B_1,\dots,B_K$ such that $B_j\cap B_k=\emptyset$ for $j\neq k$ and $\bigcup_{k=1}^K B_k=\mathcal{D}$.
The batch losses:

$L_k(\theta)=\sum_{i\in B_k}\ell(\theta;x_i)$

Then, by linearity:

$\nabla_\theta\Big(\sum_{k=1}^K L_k(\theta)\Big)
=\sum_{k=1}^K \nabla_\theta L_k(\theta)
=\sum_{k=1}^K \sum_{i\in B_k} \nabla_\theta \ell(\theta;x_i)
=\sum_{i=1}^N \nabla_\theta \ell(\theta;x_i)$

Therefore, a backward pass on the sum of losses over all disjoint batches is equivalent to GD.

This equivalence assumes the objective is a simple loss evaluated at the same parameters $\theta$.
If the forward computation depends on batch specific statistics or randomness
(e.g., BatchNorm, Dropout), then training in separate batches changes the computed function and the resulting gradient
will not match the gradient of a single full batch forward.


B) In the proposed approach we run forward one pass per batch without backpropagation, and only at the end call a single backward pass on the sum of all batch losses.

- Each forward pass produces intermediate tensors (activations) that are required to compute gradients later.
- Since no backward pass has been executed, these saved intermediates cannot be freed, because they are needed for the eventual backward call.

Therefore, even if a single batch fits in memory, the memory footprint accumulates across batches. After processing $t$ batches, we are effectively storing the activations for all $t$ forwards.

c) To solve this issue we can use gradient accumulation. For each batch, we will run a forward pass and immediately call backward(), but delay the parameter update step until we have processed all batches. This way, the computation graph of each batch can be freed right after its backward pass, so memory does not accumulate across batches.


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

1. A) Optimization error - the failure of an algorithm to find the absolute best set of parameters, often getting stuck in local minima.

B) Generalization error - the difference between a machine learning model's performance on training data versus its performance on new, unseen data, measuring how well it generalizes.

c) Approximation error - the difference between a true, exact value and its estimated or approximated value, occurring due to rounding, model limitations, or measurement inaccuracies.

2. Based on those plots:

A) Optimization error — not high

We can see that the train loss drops quickly and stayes low, and the train accuracy is ~93–94%. That means the optimizer got into a good minimum for this model.

B) Generalization error — high

We can see that the train accuracy 93–94% while test accuracy is lower and noisier (85–91%), and test loss is higher and swinging compared to train loss.

C) Approximation error - not high

The decision boundary is non-linear and can bend to complex class structure, which suggests the hypothesis class is sufficiently
expressive and the model is not underfitting. The training errors are probbably because of class overlap / noise in the data (regions where red and blue points are mixed), rather than by the model being too simple to represent the true boundary.
"""

part3_q2 = r"""
**Your answer:**
- An example scenario where we would prefer to optimize for FPR at the cost of increasing FNR: Spam filter for email

We would rather be careful about calling something “spam”, because the cost of a false positive - a real important email gets sent to spam and we will miss critical information, is much higher than the cost of a false negative - more spam slips through.

- An example scenario where we would prefer to optimize for FNR at the cost of increasing FPR: Medical screening test

Missing a true positive - a sick person is told he is healthy can cause serious harm, more than a false alarm - a healthy person get flagged and mabye has to do more tests.
"""

part3_q3 = r"""
**Your answer:**

1) Decision boundaries and model performance you obtained for the columns (fixed depth):

As width increases, the network can represent more complicated functions. We can see that the decision boundary goes from simple and almost linear to more curved and comlex, which fits the data structure better.

2) Decision boundaries and model performance you obtained for the rows (fixed width):

As depth increases, the model can build features in stages: early layers learn simple patterns, later layers combine them into more complex ones. We can see that the decision boundary becomes more expressive and structured.

3) depth=1,width=32 vs depth=4,width=8:

Even with the same total parameters, the first model learns a simpler separator that looks like underfitting to the curved structure, while the second model can build multi step features, which tends to represent curved boundaries more efficiently.

4) Selecting the threshold shifts the decision boundary without changing the learned features, trading off false positives vs false negatives. It improved the results on the test set because the data overlap region might be shaped so that moving the boundary slightly reduces a lot of errors on one side, while adding only a few on the other, leading to a “sweet spot” threshold.

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