import torch
import torch.nn.init as init
import torch.nn.functional as F

from inspect import ismethod
import math


class Tensor(torch.Tensor):
    def __init__(self):
        super(Tensor, self).__init__()

        self.source = None
        self.destinations = []

    def __init__(self, size):
        self.__init__()

        self.resize_(size)

    def set_source(self, source):
        assert self.source == source or self.source == None, "Only one source per tensor"
        self.source = source

    def add_destination(self, destination):
        self.destinations.append(destination)


class Module(object):
    def __init__(self):
        super(Module, self).__init__()

        self.sources = []
        self.destinations = {}

        self.clear_cached_io()

    def clear_cached_io(self):
        self.inputs = []
        self.outputs = []

    def to(self, device):
        # Set all tensors to be on the requested device
        for attr in dir(self):
            if isinstance(attr, torch.Tensor):
                setattr(self, attr, getattr(self, attr).to(device))

        # Set all child Modules to be on the requested device
        for attr in dir(self):
            if isinstance(attr, Module):
                getattr(self, attr).to(device)

    def requires_scaling(self):
        result = False

        for attr in dir(self):
            if isinstance(attr, Module):
                result = result or attr.requires_scaling()

        return result

    def trainable(self):
        return False

    def solved(self):
        return False

    def add_source(self, input):
        self.sources.append(input)

    def add_destination(self, idx, output):
        self.destinations[idx] = output

    def __call__(self, *args):
        assert hasattr(self, 'forward') and ismethod(
            getattr(self, 'forward')), "Forward method not available"

        need_dag = hasattr(self, 'backward') and ismethod(
            getattr(self, 'backward'))

        # Set DAG information
        if need_dag:
            source_in_history = False
            for idx, arg in enumerate(args):
                # Let this tensor know where it's being used, but only if it was an input (has no source module)
                if arg.source is None:
                    arg.add_destination((self, idx))

                    # Store this input if it has no "source" and the current node is trainable
                    if self.trainable():
                        self.inputs.append(arg)
                else:
                    source_in_history = True

                    # Add the layer source for this input
                    self.add_source(arg.source)

                    # Fetch the latest source layer, and it know which layer it feeds this output to
                    source_layer, source_output_idx = self.sources[-1]
                    source_layer.add_destination(
                        source_output_idx, (self, idx))

        # Run the forward pass
        results = self.forward(args)

        # If there's only a single output, make it a tuple of one
        results_tuple = results if hasattr(results, '__iter__') else (results,)

        if need_dag:
            # Set DAG information
            for idx, result in enumerate(results_tuple):
                # We only give a tensor a source if it has an unsolved trainable node in its history
                if (self.trainable() and not self.solved()) or source_in_history:
                    result.set_source((self, idx))

        return results

    def solve(self):
        assert len(self.destinations) == len(
            self.outputs), "Not all outputs are cached"

        inputs = self.backward(self.outputs)
        self.outputs = []

        # Send the results back to their sources
        for input, (source, idx) in zip(inputs, self.sources):
            source.destinations[idx] = input


class Activation(Module):
    def __init__(self):
        super(Activation, self).__init__()


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def requires_scaling(self):
        return True

    def forward(self, x):
        result = torch.sigmoid(x)

        if (self._high is not None) and (self._low is not None):
            result = torch.lerp(self._low, self._high, result)

        return result

    def backward(self, y):
        self._low = torch.min(y)
        self._high = torch.max(y)
        return torch.logit(y)


class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def requires_scaling(self):
        return True

    def forward(self, x):
        result = torch.tanh(x)

        if (self._high is not None) and (self._low is not None):
            result = torch.lerp(self._low, self._high, result)

        return result

    def backward(self, y):
        self._low = torch.min(y)
        self._high = torch.max(y)
        return torch.atanh(y)


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        z = torch.exp(x)
        self.total = torch.sum(z, dim=-1)

        return z / self.total

    def backward(self, y):
        z = y * self.total
        self.total = None

        return torch.log(z)


class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return torch.max(x, torch.zeros_like(x))

    def backward(self, y):
        y[y <= 0] = -torch.rand_like(y)


class LeakyRelu(Activation):
    def __init__(self, alpha=0.3):
        super(Relu, self).__init__()

        self.alpha = torch.tensor([alpha])

    def forward(self, x):
        x[x <= 0] = self.alpha * x
        return x

    def backward(self, y):
        y[y <= 0] = y / self.alpha
        return y


class Elu(Activation):
    def __init__(self):
        super(Elu, self).__init__()

    def forward(self, x):
        x[x <= 0] = torch.exp(x) - 1.0
        return x

    def backward(self, y):
        y[y <= 0] = torch.log(y) + 1.0
        return y


class Layer(Module):
    def __init__(self):
        super(Layer, self).__init__()

        self.trained = False

    def set_trained(self):
        self.trained = True


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        self.orig_shape = x.shape

        return torch.flatten(x, start_dim=1)

    def backward(self, y):
        return y.view(self.orig_shape)


class Cat(Layer):
    def __init__(self, dim=1):
        super(Cat, self).__init__()

        self.dim = dim

    def forward(self, *x):
        self.splits = [x_.shape[self.dim] for x_ in x]

        return torch.cat(x, dim=self.dim)

    def backward(self, y):
        return torch.split(y, self.splits, dim=self.dim)


class Split(Layer):
    def __init__(self, splits, dim=1):
        super(Split, self).__init__()

        self.splits = splits
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.splits, dim=self.dim)

    def backward(self, *y):
        return torch.cat(y, dim=self.dim)


class Trainable(Layer):
    def __init__(self, splits, dim=1):
        super(Trainable, self).__init__()

        self.solved = False

    def trainable(self):
        return True

    def set_solved(self):
        self.solved = True
        self.clear_cached_io()

    def solved(self):
        return self.solved


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.use_bias = bias
        self.weights = Tensor(
            size=(out_features, in_features + 1 if bias else in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x):
        if self.use_bias:
            x = torch.cat(
                (x, torch.ones(size=x.shape[:-1] + (1,))),
                dim=-1
            )
        return F.linear(self.weights, x)

    def backward(self, y):
        if self.use_bias:
            return F.linear(y[..., -1] - self.weights[-1], torch.pinverse(self.weights[:-1]))
        else:
            return F.linear(y, torch.pinverse(self.weights))

    # def update(self, x, y):

    #     if self.include_bias:
    #         x = numpy.concatenate([x, numpy.ones((x.shape[0], 1))], axis = 1)

    #     self.weight = numpy.matmul(numpy.linalg.pinv(x, rcond = RCOND), y)


class Conv2d(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size,
                 dilation=1,
                 padding=0,
                 stride=1,
                 bias=True):
        super(Conv2d, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size if not hasattr(
            kernel_size, '__iter__') else (kernel_size, kernel_size)
        self.dilation = dilation if not hasattr(
            dilation, '__iter__') else (dilation, dilation)
        self.padding = padding if not hasattr(
            padding, '__iter__') else (padding, padding)
        self.stride = stride if not hasattr(
            stride, '__iter__') else (stride, stride)

        self.area = kernel_size[0] * kernel_size[1]

        self.use_bias = bias

        self.weights = Tensor(
            size=(out_features, in_features * (self.area + 1 if bias else self.area)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x):
        assert self.input_size is None, "Conv2d modules cannot currently be shared"

        # Get the size of the input
        self.input_size = x.shape[-2:]

        x = F.unfold(x, self.kernel_size, self.dilation,
                     self.padding, self.stride)
        x.transpose_(-2, -1)

        if self.use_bias:
            x = torch.cat(
                (x, torch.ones(size=x.shape[:-1] + (self.in_features,))),
                dim=-1
            )
        x = F.linear(x, self.weights)

        # TODO review: Calculate output_size based on padding limits
        output_size = torch.floor((self.input_size + 2.0 * self.padding -
                                   self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        x.transpose_(-2, -1)
        x.view_((x.shape[0], x.shape[1],) + output_size)

        return x

    def backward(self, y):
        y.view_((y.shape[0], y.shape[1], -1))
        y.transpose_(-2, -1)

        if self.use_bias:
            y = F.linear(y[..., -1] - self.weights[-1],
                         torch.pinverse(self.weights[:-1]))
        else:
            y = F.linear(y, torch.pinverse(self.weights))

        y.transpose_(-2, -1)
        y = F.fold(y, self.input_size, self.kernel_size,
                   self.dilation, self.padding, self.stride)

        divisor = torch.ones_like(y)
        divisor = F.fold(divisor, self.input_size, self.kernel_size,
                         self.dilation, self.padding, self.stride)

        self.input_size = None

        return y / divisor
