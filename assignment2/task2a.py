import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray, mean: float, std: float):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    X = (X-mean)/std
    X = np.insert(X, X.shape[1], 1, axis=1)

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    xLoss = -np.einsum('ij,ij',targets, np.log(outputs))/(targets.shape[0])
    return xLoss


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        self.delta_wt = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            self.delta_wt.append(w.copy())
            prev = size
        # Initializing the weights according to the use_improved_weight_init variable
        self.improved_weight_init() if self.use_improved_weight_init else self.weight_init()

        # Initialize lists of gradients, z-values and a-values(output).
        self.grads = [None for i in range(len(self.ws))]
        self.z_vals = [None for i in range(len(self.ws))]
        self.a_vals = [None for i in range(len(self.ws))]



    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        # Calculating the output of each layer
        self.z_vals[0] = np.dot(X, self.ws[0])
        self.a_vals[0] = self.improved_sigmoid(self.z_vals[0]) if self.use_improved_sigmoid else self.sigmoid(self.z_vals[0])
        self.z_vals[1] = np.dot(self.a_vals[0], self.ws[1])
        self.a_vals[1] = self.softmax(self.z_vals[1])

        y = self.a_vals[1]

        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        # output layer grads
        delta_k = -(targets-outputs)
        self.grads[1] = np.dot(self.a_vals[0].T, delta_k)/(X.shape[0]) # shape (10,64)

        # hidden layer grads
        delta_j =  (np.dot(delta_k, self.ws[1].T))*self.slope_sigmoid(self.z_vals[0]) if self.use_improved_sigmoid \
                   else (np.dot(delta_k, self.ws[1].T))*(-self.a_vals[0]*(self.a_vals[0]-1))
        self.grads[0] = np.dot(X.T, delta_j)/(X.shape[0])

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]

    def update_weights(self, learning_rate):
        self.ws[0] -= learning_rate*self.grads[0]
        self.ws[1] -= learning_rate*self.grads[1]

    def update_weights_momentum(self, learning_rate:float, momentum_gamma:float):
        self.delta_wt[0] = (learning_rate*self.grads[0] + momentum_gamma*self.delta_wt[0])
        self.delta_wt[1] = (learning_rate*self.grads[1] + momentum_gamma*self.delta_wt[1])
        self.ws[0] -= self.delta_wt[0]
        self.ws[1] -= self.delta_wt[1]

    def weight_init(self):
        self.ws[0] = np.random.uniform(-1, 1, self.ws[0].shape)
        self.ws[1] = np.random.uniform(-1, 1, self.ws[1].shape)
    def improved_weight_init(self):
        self.ws[0] = np.random.normal(loc=0, scale=(1/np.sqrt(self.ws[0].shape[0])), size=self.ws[0].shape)
        self.ws[1] = np.random.normal(loc=0, scale=(1/np.sqrt(self.ws[1].shape[0])), size=self.ws[1].shape)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0/(1 + np.exp(-z))

    def improved_sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.7159 * np.tanh((2.0/3.0)*z)

    def slope_sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.7159 * 2/(3*np.cosh((2.0/3.0)*z))

    def softmax(self, z: np.ndarray) -> np.ndarray:
        return np.exp(z)/(np.sum(np.exp(z), axis=1).reshape(-1, 1))



def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # initializing an array of zeros with the right dimentions
    one_hot_Y = np.zeros((Y.shape[0], num_classes), dtype=int)
    # using arrange to set the right entries to one.
    one_hot_Y[np.arange(Y.shape[0]), Y[:,0]] = 1

    return one_hot_Y


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    # calculating mean and standard deviation of the training data.
    Xt_mean = np.mean(X_train)
    Xt_std = np.std(X_train)
    X_train = pre_process_images(X_train, Xt_mean, Xt_std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
