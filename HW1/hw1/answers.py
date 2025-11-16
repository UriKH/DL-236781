r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False. 
Not every disjoint split is equally useful. The train and test sets should be balanced and representative of the true data distribution.
For example, assume we want to train a cat image binary classifier, if the model will only be trained on the 0 labeled images (images which are not of cats), the model will not be able to recognize a cat on the test set well. The model won't be able to generalize - i.e. to learn about cat images. 

2. False.
The test set must remain untouched during cross validation. Letting the test set influence any decision will causes leakage: overfitting to the test and biased estimate of generalization.

3. True.
In cross validation, each fold’s validation performance is indeed a proxy for the model’s generalization beacuse it mimics training and then evaluating new sampels (test-like set).

4. True. 
- When talking about linear regression, injecting noise into the labels could help validate the robustness of the model by making sure it preforms well while not overfitting specific function uniqueonly  to the dataset. 
This method can help us be more confidant that the model cuptures the true data distribution from which the dataset is sampled from.
- In classification this doesn't make much sense as we want the model to learn as much as possible from the data. Swaping labels could "confuse" the model. Nonetheless, adding noise to the data (not labels) could help make the model more robust and less sensitive to changes in the input.
"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

part1_q2 = r"""
**Your answer:**
His approach isn't justified. When he chooses the value of $\lambda$ based on the test set performance, he is letting the test set influence the model's decisions.
This treatment is tipical in the case of cross-validation where instead of the test set we use a dedicated validation set (and many times we even have different "validation sets" i.e. kfold cross validaiton). In this case, that turns the test set into a singular validation set, causing overfiting on the test set and biased estimation of generalization.
"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
If we allow $\Delta < 0$ for the SVM loss $L(\mat{W})$ we penalize our model only if the wrong score exceeds the correct score by more than $|{\Delta}|$ since:
$$\text{max(0, }\Delta + {w_j}^\top x_i - {w_{y_i}}^\top x_i\text{)} > 0 \iff {w_j}^\top x_i - {w_{y_i}}^\top x_i > -\Delta \iff {w_j}^\top x_i - {w_{y_i}}^\top x_i > |\Delta|$$

In other words:

First note that ${w_j}^\top x_i$ is the score prediction for sample $x_i$ for class $j$ and ${w_{y_i}}^\top x_i$ is the score for sample $i$ to be in the correct class $y_i$. 

This means that for a good model we want ${w_j}^\top x_i < {w_{y_i}}^\top x_i \Rightarrow {w_j}^\top x_i - {w_{y_i}}^\top x_i < 0$. 

Adding a negaive constant to the aformentioned term ($\Delta < 0$) means we don't penalize for wrong classificaion in the case: $ ({w_j}^\top x_i >) \,\, {w_j}^\top x_i + \Delta > {w_{y_i}}^\top x_i$
"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

part2_q2 = r"""
**Your answer:**
The linear model learns a template for each digit \(j\): a weight vector \(w_j\) that when reshaped to a matrix, looks like the average sample of that digit. For an input image \(x\) the score for class \(j\) is the dot product \( $w_j^T x$ \), which measures how well \(x\) matches the template. Bright (positive) pixels in the weight image increase the score and dark (negative) pixels decrease it. Since the model is linear, these templates are blurry averages and the model lacks invariance to shifts or rotation.

Based on that, common classification errors come from look alike digits(e.g., \(1\) vs. \(7\), \(0\) vs. \(8\)) and sensitivity to position and writing style.
"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

part2_q3 = r"""
**Your answer:**
Based on the graph of the training set loss, we would say that the learning rate we chose is good.
The learning rate is large enough to make meaningful progress, yes small enough to produce a smooth decreasing loss with stable convergence.

Based on the graph of the training and test set accuracy, we would say that the model is slightly overfitted to the training set.
We can see that the train accuracy \( ≈ 92\% \) is consistently a bit higher than the validation accuracy \( ≈ 89\% \). That gap indicates mild overfitting, but generalization still remains good.
"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
*Your answer:*

The ideal pattern to see in a residual plot should be many points scattered around 0 on the $y-\hat{y}$ axis within a small standard devietion from it.
We would also prefer the number of outliers (i.e. far from $y-\hat{y} = 0$) to be as small as possible.

Based on the residual plots produced in top 5 features, we can say that the fitness of the trained model is pretty good but could possibly be better: most residuals are centered around 0 and within a standard deviation of about 5.

In the final plot we can see that after CV, the residuals are tighter around 0 with fewer outliers meaning the model fits the data better. Note that the standard deviation in this case is around 2.5 - two times smaller than in the other graph.

## TODO: COMPARE WITH SOMEONE - NOT SURE WE LOOKED ON THE RIGHT GRAPHS!!!!
"""

# Write your answer using *markdown* and $\LaTeX$:
# python
# # A code block
# a = 2
# 
# An equation: $e^{i\pi} -1 = 0$

part3_q2 = r"""
*Your answer:*

### Answer 2.1
After adding non-linear features to our data the regression model is still a linear model.
After adding non-linear features we get get a new feature vector: $\phi (x)$

The model prediction is given using: $ \hat{y} = w^\top \phi (x) + b $

This is linear in the sense that the function given by the regression is a hyperplane defined by $w$ and $b$ in $d = \dim(\phi(x))$ dimensions.

This could be viewed also as that for all $i$: $\hat{y_i} = w_i \cdot \phi (x)_i + b_i$ is a non-linear function of the input x.

### Answer 2.2

Yes. 
We learned in intro to ML that using the kernel trick, specifically the RBF kernel we could achieve this goal.

RBF kernel is a special function, given to sample vectors $x_i, x_j$ computes the matrix $K$ given by: $$K_{i,j} = K(x_i, x_j) = e^{-\gamma || x_j - x_i||^2 }$$
This is done using the mapping function $\phi(\textbf{x})$ which maps the vector to a new infinite feature space (which is not needed due to the kernel trick).

This feature mapping is so powerful it could allow us match any function $f(x)$ given a finite amount of points {(x_i,y_i)}_{i=1}^n such that $f(x_i)=y_i$:

Note that $K \succ 0$ (i.e. K is PD) therefore it doesn't have zero eigenvalues thus fully ranked -> invertible.

The solution for this regression problem is therefore $w = K^{-1}y$.

We can use these weights later for prediction by creating $k'(x)$ using vectors from the training set denoted $v_j$: 
$$ k'(x)_i = v_j^T x \Longrightarrow \hat{y} = w^T k'(x) $$


### Answer 2.3

A linear classification model defines a hyperplane W representing the decision boundary such that: $$ \hat{y} = \text{sign}(w^\top x + b) $$

Adding non-linear features transform the decision boundary to:
$$ \hat{y} = \text{sign}(w^\top \phi (x) + b) $$

When we are looking in the feature space (dimension of $\text{Img}(\phi(x))$), the decision boundary is still a hyperplane. But, when looking back in terms of the original input, it is no longer a hyperplane. This is what makes feature mapping so powerful.
"""

# Write your answer using *markdown* and $\LaTeX$:
# python
# # A code block
# a = 2
# 
# An equation: $e^{i\pi} -1 = 0$

part3_q3 = r"""
*Your answer:*
1. $x$ and $y$, both $\sim \text{Uniform}(0,1)$ independently, therefore the expected value is:

$
\mathbb{E}_{x,y}[|y-x|] = \int_0^1 \big( \int_0^1 |y-x| \, dx \big) \, dy 
= \int_0^1 \big( \int_0^y y-x \, dx + \int_y^1 x-y \, dx \big) \, dy 
= \int_0^1 \left( \left[ yx - \frac{x^2}{2} \right]{x=0}^{x=y}+ \left[ \frac{x^2}{2} - yx \right]{x=y}^{x=1} \right) \, dy
= \int_0^1 \left( y^2 - \frac{y^2}{2} + \frac{1}{2} - y - \left( \frac{y^2}{2} - y^2 \right) \right) \, dy
$

$
= \int_0^1 \left( \frac{y^2}{2} + \frac{1}{2} - y + \frac{y^2}{2} \right) \, dy
= \int_0^1 \left( y^2 - y + \frac{1}{2} \right) \, dy
= \left[ \frac{y^3}{3} - \frac{y^2}{2} + \frac{y}{2} \right]_0^1
= \frac{1}{3} - \frac{1}{2} + \frac{1}{2}
= \frac{1}{3}.
$


2. $x$ is a constant, therefore:

$
\mathbb{E}_x[|\hat{x}-x|] = \int_0^1 |\hat{x}-x| \, dx
= \int_0^{\hat{x}} (\hat{x}-x) \, dx + \int_{\hat{x}}^1 (x-\hat{x}) \, dx
$
$
= \left[ \hat{x}x - \frac{x^2}{2} \right]0^{\hat{x}} + \left[ \frac{x^2}{2} - \hat{x}x \right]{\hat{x}}^1
= \hat{x}^2 - \frac{\hat{x}^2}{2} + \frac{1}{2} - \hat{x} - \left( \frac{\hat{x}^2}{2} - \hat{x}^2 \right)
= \hat{x}^2 - \hat{x} + \frac{1}{2}.
$


3. We can drop the value of the scalar of the polynomial because the value of $\hat{x}$ that minimizes the polynomial is the same, regardless of adding a constant:

$
\frac{d}{d\hat{x}} \left( \hat{x}^2 - \hat{x} + \frac{1}{2} \right)
= 2\hat{x} - 1
= \frac{d}{d\hat{x}} \left( \hat{x}^2 - \hat{x} \right) \Rightarrow
$

$
\arg\min_{\hat{x}} \left( \hat{x}^2 - \hat{x} + \frac{1}{2} \right)
= \arg\min_{\hat{x}} \left( \hat{x}^2 - \hat{x} \right).
$
"""

# Write your answer using *markdown* and $\LaTeX$:
# python
# # A code block
# a = 2
# 
# An equation: $e^{i\pi} -1 = 0$

# ==============

# ==============