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

$$\Big(\pderiv{\mat{Y}}{\mat{X}}\Big)_{n,o,m,i} = \pderiv{\mat{Y}_{n,o}}{\mat{X}_{m,i}}$$

Therefore, the Jacobian tensor will have shape (64, 512, 64, 1024).

B. If we view the Jacobian $\pderiv{\mat{Y}}{\mat{X}}$ as a 2D block matrix, we will get $N \times N$ blocks of (out_features × in_features) such that the block (i,j) is the derivative of the i-th output (i-th row of Y) with respect to j-th input sample (j-th row of X). For a linear layer $Y=XW^T$, each output depends only on its own input sample. Therefore we get: 
$$
    \Big(\pderiv{\mat{Y}}{\mat{X}}\Big)_{i,j} = \begin{cases}
            W & i = j \\
            0 & i \ne j
    \end{cases}
$$
The structure is a block - diagonal matrix with N diagonal blocks, equal to $W$ and all other entries are zero.

C. Yes, apart from VJP we can use the Jacobian’s block structure. As we say previously, $\pderiv{\mat{Y}}{\mat{X}}$ is a block - diagonal matrix with N diagonal blocks, equal to $W$ and all other entries are zero. Therefore, there is no need to materialize the full Jacobian, we can just represent it by storing only $W$. The new tensor shape will be (in_features × out_features).

D. Given the gradient of the output w.r.t. some downstream scalar loss $L$, $\delta\mat{Y} := \pderiv{L}{\mat{Y}}$ we can use the chain rule to calculate the downstream gradient w.r.t. the input ($\delta\mat{X}$) without materializing the Jacobian. Denote the $k$-th example by:

$$
X_k \in \mathbb{R}^{1 \times in\_features}, \qquad
Y_k \in \mathbb{R}^{1 \times out\_features}, \qquad
Y_k = X_k W^\top
$$

Therefore:

$$ \delta X_k
:= \frac{\partial L}{\partial X_k}
= \frac{\partial L}{\partial Y_k}\,\frac{\partial Y_k}{\partial X_k} = \delta Y_k W
$$

Stacking all examples gives:

$$ \boxed{
\delta X := \frac{\partial L}{\partial X} = \delta Y\, W
} $$

E. W has shape (512, 1024) and Y has shape (64, 512), The Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$ describes, for each output element, how it changes with respect to each weight element:

$$\Big(\pderiv{\mat{Y}}{\mat{W}}\Big)_{n,o,p,i} = \pderiv{\mat{Y}_{n,o}}{\mat{W}_{p,i}}$$

Therefore, the Jacobian tensor will have shape (64, 512, 512, 1024).  If we view the Jacobian $\pderiv{\mat{Y}}{\mat{W}}$ as a 2D block matrix, we will get
$N \times \text{out\_features}$ blocks of size
$(\text{out\_features} \times \text{in\_features})$ such that the block $(i,j)$ is the derivative
of the $i$-th output sample with respect to the $j$-th row of $W$. 
"""

part1_q2 = r"""
**Your answer:**

Yes, second order derivatives (Hessian) can be helpful for optimization (but are not commonly used as it is quite expensive to compute unlike the gradient).

The second order information is useful when the condition number is high because it tells us how the slope changes in different directions.
If the condition number is high this means that in some directions the optimization landscape is much more steep. 

This means we would like to take very big learning rate in order to overcome the low steepness in one direction but a very small learning rate to overcome the high steepness in the other direction. 
Second order derivatives address this by rescaling the update using curvature. For example, as we learned in the lectures Newton’s method uses:

$$ \theta_{t+1} = \theta_t - H^{-1} \nabla L (\theta_t) $$

which effectively normalizes the step size differently along different directions, often leading to much faster convergence.
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

$$p_1 = (0.49, 0.51)$$

$$p_2 = (0.49, 0.51)$$

$$\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - (0,1)^T \cdot log(0.51, 0.51) = -log(0.51) = 0.673$$

Epoch 2 – 50% accuracy:

$$p_1 = (0.01, 0.99)$$

$$p_2 = (0.51, 0.49)$$

$$\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - (0,1)^T \cdot log(0.49, 0.99) = -log(0.99) = 0.01$$


In the first epoch both classifications are correct but it is possible that the gradient is big enough for the first sample and much smaller for the second one such that the updated model will enforce correct classification on the first sample on the count of the second.
"""

part2_q3 = r"""
**Your answer:**
1. Similarities of gradient descent (GD) and stochastic gradient descent (SGD):

- Both aim to minimize the same loss function

- Same update rule $\theta \leftarrow \theta - \eta \cdot gradient$

Differences of gradient descent (GD) and stochastic gradient descent (SGD):

- How they compute the gradient - GD uses all data while SGD pick one sample from a uniform distrubution on tha dataset.

- Cost - GD goes through all N examples to compute one gradient while SGD computes the gradient w.r.t only one sample, so each step is much faster.

- Noise - GD get deterministic path and the loss usually decreases smoothly while in SGD each gradient is noisy (since it is computed on different samples each epoch) and the randomness in the sampling gives different paths.

2. Yes, momentum can be used with GD. Momentum can accelerate convergence by accumulating velocity in directions with consistent gradients often leading to faster and stable convergence. The main difference from SGD is that in GD momentum is not primarily “noise smoothing”.

3.  A) Let the dataset be $\mathcal{D}=\{x_i\}_{i=1}^N$ and the loss be $\ell(\theta; x_i)$.

$$L(\theta)=\sum_{i=1}^N \ell(\theta;x_i)$$

hence its gradient is:

$$\nabla_\theta L(\theta)=\sum_{i=1}^N \nabla_\theta \ell(\theta;x_i)$$

Assume we partition the dataset into disjoint batches $B_1,\dots,B_K$ such that $B_j\cap B_k=\emptyset$ for $j\neq k$ and $\bigcup_{k=1}^K B_k=\mathcal{D}$.
The batch losses:

$$L_k(\theta)=\sum_{i\in B_k}\ell(\theta;x_i)$$

Then, by linearity:

$$\nabla_\theta\Big(\sum_{k=1}^K L_k(\theta)\Big)
=\sum_{k=1}^K \nabla_\theta L_k(\theta)
=\sum_{k=1}^K \sum_{i\in B_k} \nabla_\theta \ell(\theta;x_i)
=\sum_{i=1}^N \nabla_\theta \ell(\theta;x_i)$$

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

4) Selecting the threshold shifts the decision boundary without changing the learned features, trading off false positives vs false negatives. 
By default, the boundary is at 0.5 probability which can cause a lot of misclassifications, but moving it can better align with the data distribution. 
In our case, it did improve the results on the test set because the data overlap region might be shaped so that moving the boundary slightly reduces a lot of errors on one side, while adding only a few on the other, leading to a “sweet spot” threshold.

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

Differences between the regular block and the bottleneck block in terms of:

1) Number of parameters:

Let $K_{width}, K_{height}$ be the kernel dimensions of a convolution, and let $C_{in}, C_{out}$ be the number of input and output channels. Then the number of weight parameters in a convolution layer is give by: $ \# params = K_{width} \cdot K_{height} \cdot C_{in} \cdot C_{out}$

- In regular block each convolution has kernel $3 \times 3$, with $C_{in} = C_{out} = 256$ Therefore:

$$ \# params_{regular} = 2 \cdot (3 \cdot 3 \cdot 256 \cdot 256) = 1179648$$

- The bottleneck consists of: $1 \times 1$ convolution with $C_{in} = 64, C_{out} = 256$, after $3 \times 3$ convolution with $C_{in} = C_{out} = 64$, after $1 \times 1$ convolution with $C_{in} = 256, C_{out} = 64$.

$$ \# params_{bottleneck} = (1 \cdot 1 \cdot 256 \cdot 64) + (3 \cdot 3 \cdot 64 \cdot 64) + (1 \cdot 1 \cdot 64 \cdot 256) = 69632$$

Overall we get:

$$ \frac{\# params_{regular}}{\# params_{bottleneck}} = \frac{1179648}{69632} = 16.941$$

2) Number of floating point operations required to compute an output (qualitative assessment):

Compute is proportional to: $ \# flops = H \cdot W \cdot K_{width} \cdot K_{height} \cdot C_{in} \cdot C_{out}$. Therefore:

$$ \frac{\# flops_{regular}}{\# flops_{bottleneck}} = \frac{H \cdot W \cdot 1179648}{H \cdot W \cdot 69632} = 16.941$$

3) Ability to combine the input: 

(1) spatially (within feature maps)
- Regular block: has two $3 \times 3$ convolutions, giving greater spatial processing depth (two spatial mixing steps).
- Bottleneck block: has only one $3 \times 3$ convolution, so it has less spatial mixing depth per block.

(2) across feature maps.
- Regular block: also mixes channels, but this mixing happens within the $3 \times 3$ convolutions.
- Bottleneck block: explicitly uses $1 \times 1$ convolutions to perform efficient channel mixing. 
"""


part4_q2 = r"""
**Your answer:**

Given $M$ a $m \times n$ matrix with small entries meaning $ \forall i,j  ; |M_{i,j}| < 1 $.

1) Given $y_1 = M \cdot x_1$, $\frac{\partial L}{\partial y_1}$, we can use the chain rule to derive:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial y_1} \frac{\partial y_1}{\partial x_1} = M \frac{\partial L}{\partial y_1}$$.


2) Given $ y_2 = x_2 + M \cdot x_2 $, $ \frac{\partial L}{\partial y_2} $, we can use the chain rule to derive:

$$ \frac{\partial L}{\partial x_2} = \frac{\partial L}{\partial y_2} \frac{\partial y_2}{\partial x_2} = (M + I) \frac{\partial L}{\partial y_2}$$.

3) Assume each layer is $x^{(t)} = M_t\,x^{(t-1)}$.

By the chain rule,

$$\frac{\partial L}{\partial x^{(t-1)}} = \frac{\partial x^{(t)}}{\partial x^{(t-1)}} \frac{\partial L}{\partial x^{(t)}} = M_t^{\top}\,\frac{\partial L}{\partial x^{(t)}}$$.

Applying this repeatedly over $k$ layers gives

$$\frac{\partial L}{\partial x^{(0)}} = \left(\prod_{t=1}^{k} M_t\right)\frac{\partial L}{\partial x^{(k)}}$$.

If the matrices $M_t$ have small entries, multiplying many of them leading to vanishing gradients.


On the other hand with skip:

$$x^{(t)} = x^{(t-1)} + M_t\,x^{(t-1)} = (I+M_t)\,x^{(t-1)}$$.

$$\frac{\partial L}{\partial x^{(t-1)}} = \frac{\partial x^{(t)}}{\partial x^{(t-1)}} \frac{\partial L}{\partial x^{(t)}} = (I+M_t)\,\frac{\partial L}{\partial x^{(t)}}$$.

Over $k$ layers,

$$\frac{\partial L}{\partial x^{(0)}} = \left(\prod_{t=1}^{k} (I+M_t)\right)\frac{\partial L}{\partial x^{(k)}}$$.

When $M_t$ have small entries, $(I+M_t)$ is close to the identity, so the product is much less likely to shrink the gradient.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

As L increases from 2 to 4, both train and test accuracy improve. This shows that adding depth helps the model learn more expressive features. 
However, when L increases further to 8 or 16, accuracy collapses to ~10% and the loss barely improves, meaning too deep networks fail to train.

As seen in the results of the train and test, we can see the following phenomena:

1. The shallow models L=2,4: The models learned successfully. In both K=32 and K=64 experiments, the networks with L=2 and L=4 show a steady decrease in training loss and a clear increase in training accuracy. 
However, it seems that L=2 model was two shallow and didn't reach test and train results as good as L=4. 
This makes sense because with fewer layers, the model is less capable of learning complex features from the data.

2. The deeper models L=8,16: The models couldn't train, both training and test accuracy stay around ~10%, which is approximately random guessing for a 10-class problem. 

The best results come from L=4 (in both K=32 and K=64 runs). L=4 is a sweet spot because it is deep enough to learn complex features, but not so deep that training becomes unstable.

There were values of L for which the network wasn't trainable, for L=8,16. The main cause is vanishing gradients. 
As the network depth increases, gradients can become very small during backpropagation, making it difficult for the weights in earlier layers to update.

To resolve this issue we could have used:

1) Residuals - a way to prevent vanishing gradients (de facto allowing the model to remember the input sample in each layer).

2) Normalization (e.g., BatchNorm) after convolutional layers - this stabilizes activations and gradients, usually enabling convergence in deeper models.
"""

part5_q2 = r"""
**Your answer:**

In experiment 1.2 we kept the same training hyperparameters as in experiment 1.1 
and focused on the effect of width, by changing the number of filters per layer (K) while keeping the depth (L) fixed.

As seen in the results of the train and test, we can see the following phenomena:

1. For L=2, all three (K=32,64,128) trainable, both train and test loss decreased smoothly and accuracy increased steadily. 
However, the gap between different values of K was relatively small, which suggests that when the network is shallow, increasing width gives only limited benefit.

2. For L=4, increasing K had a much clearer positive effect. All models trained well, but the widest network (K=128) reached the highest test accuracy.
This indicates that if the network has enough depth to learn meaningful features, adding more channels helps because each layer can 
capture more complex patterns, which improves generalization.

3. For L=8, the network was not trainable for any value of K and the training and test accuracy stayed around 10%. 
This again points to vanishing gradients as the main issue with deep networks.

The best results come from K = 128, which reaches the highest test accuracy (about 72–73%) among the trainable settings.

Comparing experiment 1.2 to experiment 1.1, the key difference is that depth had a much stronger and more critical effect than width,
we also can see that when we changed K (for models that were trainable) the gains were moderate. For overlaping experiments between 1.1 and 1.2, results were similar.
For the failing deep case (L=8), the same remedies suggested in experiment 1.1 would apply here as well, 
such as adding residuals or normalization layers, which mitigate vanishing gradiens and therefore make training deep architectures much more feasible.

"""

part5_q3 = r"""
**Your answer:**

In experiment 1.3 we examined the effect of the number of convolutional filters in each layer.

The results show that L=2 and L=3 are trainable with this wide configuration. 
In both settings, we observe signs of overfitting in the later stages of training. 
While the training loss continues to decrease and the training accuracy keeps rising, 
the test loss reaches a minimum and then starts to increase, and the test accuracy stops improving.
The overfitting behavior is more pronounced for L=2.
L=3 generalizes better overall, which suggests that the additional depth helps learn more robust representations even though 
some overfitting still appears.

In contrast, L=4 is not trainable in this setup: both training and test accuracy remain around 10%.
This failure indicates that with more depth (and already large width), gradientscan become poorly scaled, making it hard for the model to learn.
As a result, the model stays close to its random guessing performance.

Overall, the best configuration in this experiment is K=\[64,128\] with L=3, achieving the highest and most stable test accuracy among the trainable models.

"""

part5_q4 = r"""
**Your answer:**

In experiment 1.4 we switched from CNN model type to ResNet and tested deeper configurations. 

The first plot set compares L=8, 16, 32 for K=32. We can see that, L=8 and L=16 are trainable: training loss keeps dropping and training accuracy climbs to ~80-90%, while test accuracy rises to the ~60. 
This is an important difference from experiment 1.1, where deeper CNNs (L $\ge$ 8) collapsed to ~10% accuracy. 
However, L=32 performs much worse: it learns slowly, reaches only ~50% train and test accuracies. 
This suggests that ResNet arcitecture helps mitigate vanishing gradients and enables training of deeper network.

The second plot set compares L=2, 4, 8 for K=\[64,128,256\]. 
Here, L=2 and L=4 are trainable, and L=4 is best on test (around ~70–71%) while L=2 reaches a lower test accuracy and shows stronger signs of overfitting
We figure that the surprising part is that L=8 again collapses to ~10% (not trainable), even though this is a ResNet.
This suggests that skip connections help a lot, 
but they don’t guarantee stability when the model is very deep and wide (many parameters, harder optimization).

Compared to experiment 1.1, the main diference is trainability at higher depth. 
In 1.1, depth L=8 and L=16 caused collapse, while in 1.4 the ResNet successfully trains them and reaches reasonable test accuracy. 
This supports the idea that residual connections can make deeper networks easier to optimize, even if the final accuracy is still in a similar range.

Compared to experiment 1.3, a similar pattern appears. In 1.3, increasing capacity and depth caused training to collapse earlier 
(L=4), while in 1.4 the ResNet can still train L=4 even with a wider configuration (K=\[64,128,256\]). 
However, pushing depth further to L=8 with the wide ResNet still fails.

Overall, experiment 1.4 suggests that ResNets improve trainability, allowing deeper models to learn where CNN fails.
Note that the training of experiments 1.1-1.3 where trained on very similar hyperparameters 1.4 was trained on different hyperparameters so the results are not fully compareable.
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

We chose three images:
1. Blurry image of a dog: The model did detect the dog but also detected his tongue as a freesbe. 
This is probably due to the fact that the image is blurry and thus the model was unable to extract the exact features.
2. A busy street: In this image we can see that although most of the faces are in high resolution and not blurred at all, due to the fact that a part of the face is occluded by another person, the model failed to classify the part of the person with only some of their face appearing.
Noteably the people in the left part of the image which are not classified as anything.
This demonstrates the occlusion pitfall in object detection.
3. A set of chiuaua images compared to cupcakes (we intentionally chose an image containing many examples to demonstrate the issue but we could have off course cut only one bad example :) ). 
In this image we can see that the model, in some contexts, classified cupckaes as dogs (clear example in the top right) or as a teddy bear apart from one case (bottom left).
This is an example of model bias - e.g. the model probably learned that teddy bears are yellow-ish and could be difformed in some ways due to their fluffy nature :).
Thus the model miss-classified the cakes altough it did know what a cake is (there is a correct classification) but was clearlly biased towards the teddy bear (or chiuaua in some cases).


In general it seems that the model preformed quite well only on the first and last images as it failed spectacularly on the second one. 
"""

part6_bonus = r"""
**Your answer:**

1. For the firs image of the blurry dog we used image sharpenin kernel in order to manipulate the image so that the model will recognize features better.
As we can see the model no longer detected the tongue as a freesbe and only detected the dog correctly.

2. For the second image of the busy street we tried using the fact that YOLO uses the sliding window approach and cut the parts it didn't work well on and check those seperately.
This approche worked quite well as we can see the model was able to detect most of the people it didn't detect in the original image.

3. For the third image of the chiuaua vs cupcakes we tried to use the samne approch as in the second image. It seems that the model was able to detect most of the cupcakes in the image even though it misclassified them as donuts, which is better than classifying them as dogs or teddy bears.

"""