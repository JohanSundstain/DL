import numpy as np

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        self.stride = stride
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
		
        fan_in = in_channels * kernel * kernel
        self.W = np.random.randn(fan_in, out_channels) * np.sqrt(2 / fan_in)
        self.input_matrix = None
        self.input_shape = None
        self.h_out = None
        self.w_out = None

    def forward(self, input):
        c, h, w = input.shape
        if self.padding > 0:
            input_padded = np.pad(
                input,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0)
        else:
            input_padded = input

        _, h_padded, w_padded = input_padded.shape
        h_out = (h_padded - self.kernel) // self.stride + 1
        w_out = (w_padded - self.kernel) // self.stride + 1

        patches = []
        for i in range(0, h_padded - self.kernel + 1, self.stride):
            for j in range(0, w_padded - self.kernel + 1, self.stride):
                patch = input_padded[:, i:i + self.kernel, j:j + self.kernel].flatten()
                patches.append(patch)

        self.input_matrix = np.stack(patches)
        res = self.input_matrix @ self.W
        output = np.stack([res[:, i].reshape(h_out, w_out) for i in range(self.out_channels)])
        self.input_shape = input.shape
        self.h_out = h_out
        self.w_out = w_out
        return output

    def backward(self, grad_output):
        num_patches = self.h_out * self.w_out
        grad_res = np.column_stack([grad_output[i, :, :].flatten() for i in range(self.out_channels)])
        self.grad_W = self.input_matrix.T @ grad_res
        grad_input_matrix = grad_res @ self.W.T

        c, h, w = self.input_shape
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding
        grad_input_padded = np.zeros((c, h_padded, w_padded))
        k = 0
        for i in range(0, h_padded - self.kernel + 1, self.stride):
            for j in range(0, w_padded - self.kernel + 1, self.stride):
                grad_patch = grad_input_matrix[k, :].reshape(self.in_channels, self.kernel, self.kernel)
                grad_input_padded[:, i:i+self.kernel, j:j+self.kernel] += grad_patch
                k += 1

        if self.padding > 0:
            grad_input = grad_input_padded[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded
        return grad_input

class Dense:
    def __init__(self, in_channels, out_channels):
        self.W = np.random.randn(in_channels, out_channels) * np.sqrt(2 / in_channels)
        self.b = np.zeros(out_channels)
        self.input = None

    def forward(self, input):
        self.input = input
        return input @ self.W + self.b

    def backward(self, grad_output):
        grad_input = grad_output @ self.W.T
        self.grad_W = np.outer(self.input, grad_output)
        self.grad_b = grad_output
        return grad_input

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * self.mask

def softmax(x):
    e_x = np.exp(x - np.max(x, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, labels):
    prop = softmax(logits)
    epsilon = 1e-15
    prop = np.clip(prop, epsilon, None)
    return -np.sum(labels * np.log(prop))