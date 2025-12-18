import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        in_channels = self.in_size[0]
        for i, c in enumerate(self.channels):
            layers += [
                nn.Conv2d(**self.conv_params, in_channels = in_channels, out_channels = c), 
                ACTIVATIONS[self.activation_type](**self.activation_params)
            ]
            if (i + 1) % self.pool_every == 0:
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
            in_channels = c

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            dummy_input = torch.zeros(1, *self.in_size)
            with torch.no_grad():
                features = self.feature_extractor(dummy_input)
            return features.view(1, -1).shape[1]
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        activation = ACTIVATIONS[self.activation_type](**self.activation_params)
        mlp = MLP(
            in_dim = self._n_features(),
            dims = self.hidden_dims + [self.out_classes],
            nonlins = [activation]*len(self.hidden_dims) + [nn.Identity()]
        )
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        feats =  self.feature_extractor(x)
        feats_flat = feats.view(feats.size(0), -1)
        out = self.mlp(feats_flat)
        # ========================
        return out

class BasicConv2d(nn.Module):
    """
    Basic Conolution 2D module 
    """
    def __init__(self, in_channels, out_channels, activation='relu', **kwargs):
        """
        Constractor for the basic convolution layer
        :param in_channels: the number of input channels to the layer
        :param out_channels: the number of output channels of the layer
        :param activation: the activation layer after the convolution
        """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Linear()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
            
class InceptionResNetBlock(nn.Module):
    def __init__(self, in_channels, branch_a, branch_b=None, branch_c=None, out_concat=128, scale=True):
        super().__init__()
        
        self.branch_a = nn.Sequential(*self.create_layers(in_channels, branch_a))
        self.branch_b = nn.Sequential(*self.create_layers(in_channels, branch_b)) if branch_b else None
        self.branch_c = nn.Sequential(*self.create_layers(in_channels, branch_c)) if branch_c else None

        # input_concat = branch_a[-1][1] + branch_b[-1][1] if branch_b else 0 + branch_c[-1][1] if branch_c else 0
        input_concat = branch_a[-1][1]
        if branch_b:
            input_concat += branch_b[-1][1]
        if branch_c:
            input_concat += branch_c[-1][1]

        self.concat_branch1x1 = BasicConv2d(input_concat, out_concat, kernel_size=1, padding='same', activation='relu')
        self.scale = scale
        # self.scale_res = Lambda(lambda x: x * 0.1)
        self.bn = nn.BatchNorm2d(in_channels)
    
    def create_layers(self, in_channels, branch_tuples):
        last = in_channels
        layers = []
        for i, (kernel_size, channels) in enumerate(branch_tuples):
            layers.append(BasicConv2d(last, channels, kernel_size=kernel_size, padding='same'))
            last = channels
        return layers

    def forward(self, x):
        branch_a = self.branch_a(x)
        branch_b = self.branch_b(x)
        branch_c = self.branch_c(x)

        outputs = [branch_a, branch_b, branch_c]
        mixed = torch.cat(outputs, 1)   # concatenation of the a, b, c branches

        mixed = self.concat_branch1x1(mixed)    # convolution on the concatenated branch
        # if self.scale:
        #     mixed = self.scale_res(mixed)   # scale down the mixed branches impact
        x = x.add(mixed)
        
        x = nn.ReLU()(self.bn(x))
        return x
        
class YourCNN(CNN):    
    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        self.batchnorm = True
        self.dropout = 0.4
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, conv_params,
            activation_type, activation_params, pooling_type, pooling_params
        )
        
        dims = [4096, 2048, 1024]
        self.mlp = MLP(
            in_dim = self._n_features(),
            dims = dims + [self.out_classes],
            nonlins = ['relu'] * len(dims) + [nn.Identity()]
        )

    def _make_feature_extractor(self):
        in_channels, _, _ = tuple(self.in_size)

        layers = []
        block_in_channels = in_channels
        conv_channels = []
        
        for i, c in enumerate(self.channels):
            conv_channels.append(c)
        
            end_of_block = ((i + 1) % self.pool_every == 0) or ((i + 1) == len(self.channels))
            if end_of_block:
                layers.append(
                    ResidualBlock(
                        in_channels=block_in_channels,
                        channels=conv_channels,
                        kernel_sizes=[3] * len(conv_channels),
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params,
                    )
                )
        
                block_in_channels = conv_channels[-1]
                conv_channels = []
        
                if (i + 1) % self.pool_every == 0:
                    layers.append(POOLINGS[self.pooling_type](kernel_size=2, stride=2))
        # ========================
        layers += [InceptionResNetBlock(block_in_channels, [(1, 64), (3, 64), (3, 64)], [(1, 32)], [(3,32), (3,32)], 32)]
        seq = nn.Sequential(*layers)
        return seq


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======       
        channels = [in_channels] + channels
        layers = []
        for i, k in enumerate(kernel_sizes[:-1]):
            layers += [nn.Conv2d(channels[i], channels[i+1], k, padding=(k // 2), bias=True)]
            if dropout > 0:
                layers += [nn.Dropout2d(dropout)]
            if batchnorm:
                layers += [nn.BatchNorm2d(channels[i+1])]
            layers += [ACTIVATIONS[activation_type](**activation_params)]
        layers += [nn.Conv2d(channels[-2], channels[-1], kernel_sizes[-1], padding=(kernel_sizes[-1] // 2), bias=True)]
        
        self.main_path = nn.Sequential(*layers)

        out_channels = channels[-1]
        if in_channels != out_channels:
            self.shortcut_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut_path = nn.Identity()
        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        super().__init__(
            in_out_channels,
            [inner_channels[0]] + list(inner_channels) + [in_out_channels],
            [1] + list(inner_kernel_sizes) + [1],
            **kwargs
        )
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        block_in_channels = in_channels
        conv_channels = []
        
        for i, c in enumerate(self.channels):
            conv_channels.append(c)
        
            end_of_block = ((i + 1) % self.pool_every == 0) or ((i + 1) == len(self.channels))
            if end_of_block:
                use_bottleneck = (
                    self.bottleneck
                    and (block_in_channels == conv_channels[-1])
                    and (len(conv_channels) >= 3)
                    and ((i + 1) % self.pool_every == 0)
                )
        
                if use_bottleneck:
                    inner_channels = conv_channels[1:-1] 
                    inner_kernels = [3] * len(inner_channels)
        
                    layers.append(
                        ResidualBottleneckBlock(
                            in_out_channels=block_in_channels,
                            inner_channels=inner_channels,
                            inner_kernel_sizes=inner_kernels,
                            batchnorm=self.batchnorm,
                            dropout=self.dropout,
                            activation_type=self.activation_type,
                            activation_params=self.activation_params,
                        )
                    )
                else:
                    layers.append(
                        ResidualBlock(
                            in_channels=block_in_channels,
                            channels=conv_channels,
                            kernel_sizes=[3] * len(conv_channels),
                            batchnorm=self.batchnorm,
                            dropout=self.dropout,
                            activation_type=self.activation_type,
                            activation_params=self.activation_params,
                        )
                    )
        
                block_in_channels = conv_channels[-1]
                conv_channels = []
        
                if (i + 1) % self.pool_every == 0:
                    layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

