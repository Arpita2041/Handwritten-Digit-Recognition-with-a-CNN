"""A small convolutional network and Adam implemented directly in NumPy."""
import numpy as np

class Conv2D:
    def __init__(self, input_channels, output_channels, rng, kernel_size=3):
        self.kernel_size = kernel_size
        self.weight = rng.normal(0, np.sqrt(2 / (input_channels * kernel_size ** 2)),
                                 (output_channels, input_channels, kernel_size, kernel_size))
        self.bias = np.zeros(output_channels)

    def forward(self, x):
        k, p = self.kernel_size, self.kernel_size // 2
        self.x_shape = x.shape
        self.padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        self.windows = np.lib.stride_tricks.sliding_window_view(
            self.padded, (k, k), axis=(2, 3))
        return np.einsum("nchwkl,ockl->nohw", self.windows, self.weight,
                         optimize=True) + self.bias[None, :, None, None]

    def backward(self, gradient):
        self.dweight = np.einsum("nohw,nchwkl->ockl", gradient, self.windows,
                                optimize=True)
        self.dbias = gradient.sum(axis=(0, 2, 3))
        dx = np.zeros_like(self.padded)
        h, w = gradient.shape[2:]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dx[:, :, i:i+h, j:j+w] += np.einsum(
                    "nohw,oc->nchw", gradient, self.weight[:, :, i, j], optimize=True)
        p = self.kernel_size // 2
        return dx[:, :, p:-p, p:-p] if p else dx

class Dense:
    def __init__(self, inputs, outputs, rng):
        self.weight = rng.normal(0, np.sqrt(2 / inputs), (inputs, outputs))
        self.bias = np.zeros(outputs)

    def forward(self, x):
        self.x = x
        return x @ self.weight + self.bias

    def backward(self, gradient):
        self.dweight = self.x.T @ gradient
        self.dbias = gradient.sum(axis=0)
        return gradient @ self.weight.T

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, gradient):
        return gradient * self.mask

class MaxPool2D:
    def forward(self, x):
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("MaxPool2D requires even height and width.")
        self.shape = x.shape
        windows = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5)
        windows = windows.reshape(n, c, h // 2, w // 2, 4)
        self.argmax = windows.argmax(axis=-1)
        return windows.max(axis=-1)

    def backward(self, gradient):
        n, c, h, w = self.shape
        windows = np.zeros((n, c, h // 2, w // 2, 4))
        np.put_along_axis(windows, self.argmax[..., None], gradient[..., None], axis=-1)
        return windows.reshape(n, c, h // 2, w // 2, 2, 2).transpose(
            0, 1, 2, 4, 3, 5).reshape(self.shape)

class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, gradient):
        return gradient.reshape(self.shape)

def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(logits, y):
    p = softmax(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1))
    loss = np.mean(log_z - shifted[np.arange(len(y)), y])
    gradient = p.copy()
    gradient[np.arange(len(y)), y] -= 1
    return float(loss), gradient / len(y)

class CNN:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.layers = [Conv2D(1, 8, rng), ReLU(), MaxPool2D(),
                       Conv2D(8, 16, rng), ReLU(), MaxPool2D(),
                       Flatten(), Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weight"):
                for name in ("weight", "bias"):
                    yield f"{i}_{name}", getattr(layer, name), getattr(layer, "d" + name, None)

    def state(self):
        return {name: value.copy() for name, value, _ in self.parameters()}

    def restore(self, state):
        expected = self.state()
        if set(state) != set(expected):
            raise ValueError("Unexpected CNN checkpoint keys.")
        for name, value, _ in self.parameters():
            if state[name].shape != value.shape:
                raise ValueError(f"Wrong checkpoint shape for {name}.")
            value[:] = state[name]

    def logits(self, x, batch_size=256):
        return np.concatenate([self.forward(x[i:i+batch_size])
                               for i in range(0, len(x), batch_size)])

    def save(self, path):
        np.savez_compressed(path, **self.state())

    @classmethod
    def load(cls, path):
        network = cls(0)
        with np.load(path, allow_pickle=False) as archive:
            network.restore({name: archive[name] for name in archive.files})
        return network

class Adam:
    def __init__(self, learning_rate=.003, weight_decay=1e-4):
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.m, self.v, self.step_number = {}, {}, 0

    def step(self, network):
        self.step_number += 1
        for name, parameter, gradient in network.parameters():
            if gradient is None:
                raise RuntimeError("Call backward before Adam.step.")
            g = gradient + (self.weight_decay * parameter if name.endswith("weight") else 0)
            self.m.setdefault(name, np.zeros_like(parameter))
            self.v.setdefault(name, np.zeros_like(parameter))
            self.m[name] = .9 * self.m[name] + .1 * g
            self.v[name] = .999 * self.v[name] + .001 * g * g
            m_hat = self.m[name] / (1 - .9 ** self.step_number)
            v_hat = self.v[name] / (1 - .999 ** self.step_number)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
