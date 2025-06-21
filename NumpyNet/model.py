from pychort import Dense, Conv2d, ReLU, softmax

class NumNet:
    def __init__(self, num_classes):
        self.first_layer = Conv2d(in_channels=3, out_channels=32, kernel=3, stride=1)
        self.relu1 = ReLU()
        self.second_layer = Conv2d(in_channels=32, out_channels=64, kernel=3, stride=2)
        self.relu2 = ReLU()
        self.third_layer = Conv2d(in_channels=64, out_channels=128, kernel=3, stride=2)
        self.relu3 = ReLU()
        self.last_layer = Dense(4608, num_classes)
        self.flatten_shape = None

    def forward(self, x):
        x = self.first_layer.forward(x)
        x = self.relu1.forward(x)
        x = self.second_layer.forward(x)
        x = self.relu2.forward(x)
        x = self.third_layer.forward(x)
        x = self.relu3.forward(x)
        self.flatten_shape = x.shape
        x = x.flatten()
        x = self.last_layer.forward(x)
        return x

    def backward(self, grad_output):
        grad = self.last_layer.backward(grad_output)
        grad = grad.reshape(self.flatten_shape)
        grad = self.relu3.backward(grad)
        grad = self.third_layer.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.second_layer.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.first_layer.backward(grad)
        return grad

    def update_weights(self, learning_rate):
        for layer in [self.first_layer, self.second_layer, self.third_layer, self.last_layer]:
            if hasattr(layer, 'W'):
                layer.W -= learning_rate * layer.grad_W
            if hasattr(layer, 'b'):
                layer.b -= learning_rate * layer.grad_b
