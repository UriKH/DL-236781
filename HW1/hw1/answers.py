r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. false. 
Not every disjoint split is equally useful. The train and test sets should be balanced and representative of the true data distribution.
For example, if a training set contains only cat images while the test set includes both cats and dogs the model wont learn the “dog” class, since we cannot learn what we dont see and will fail to generalize.

2. false.
The test set must remain untouched during cross validation. Letting the test set influences any decision will causes leakage, overfitting to the test and biased estimate of generalization.

3. true.
In cross validation, each fold’s validation performance is indeed a proxy for the model’s generalization beacuse it mimics training and evaluating new sampels.
4. false.
"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

part1_q2 = r"""
**Your answer:**
His approach isnt justified. When he chooses the value of $\lambda$ based on the test set performance, he is letting the test set influence the model decisions. That turns the test set into the validation set, causing data leakage and biased estimate of generalization.
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
$\text{max(0, }\Delta + s{ij} - s{iy_i}\text{)} > 0 \iff s_{ij} - s_{iy_i} > -\Delta \iff s_{ij} - s_{iy_i} > |\Delta|$
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
**Your answer:**

"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

part3_q2 = r"""
**Your answer:**

"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

part3_q3 = r"""
**Your answer:**

"""

# Write your answer using **markdown** and $\LaTeX$:
# ```python
# # A code block
# a = 2
# ```
# An equation: $e^{i\pi} -1 = 0$

# ==============

# ==============