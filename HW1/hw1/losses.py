import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        rows = torch.arange(x_scores.size(0))
        M = torch.clamp(x_scores - x_scores[rows, y].unsqueeze(1) + self.delta, min=0)
        M[rows, y] = 0
        loss = 1 / y.size(0) * torch.sum(M)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['M'] = M
        self.grad_ctx['y'] = y
        self.grad_ctx['x'] = x
        # raise NotImplementedError()
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        M = self.grad_ctx['M'] # (N, C)
        y = self.grad_ctx['y'] # (N,)
        x = self.grad_ctx['x'] # (N, D)

        j_ne_yi = torch.ones(M.shape, dtype=torch.int32)
        j_ne_yi[torch.arange(j_ne_yi.size(0)), y] = 0
        j_ne_yi = (j_ne_yi & (M > 0)).float()
        
        j_e_yi = torch.zeros(M.shape, dtype=torch.float32)
        j_e_yi[torch.arange(j_e_yi.size(0)), y] = -torch.sum((M > 0), dim=1).float()
        G = 1 / y.size(0) * (j_e_yi + j_ne_yi)
        
        grad = x.T @ G
        # ========================

        return grad
