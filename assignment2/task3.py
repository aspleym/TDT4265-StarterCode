import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    accuracy = 0.0
    logits = model.forward(X)

    X_hits_index = np.argmax(logits, axis=1)
    Y_hits_index = np.argmax(targets, axis=1)

    accuracy = np.count_nonzero(Y_hits_index == X_hits_index)/X.shape[0]
    return accuracy


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    n_last_val_losses = np.ones(10)

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # forward-pass, backward-pass and update-step
            logits = model.forward(X_batch)
            model.backward(X_batch, logits, Y_batch)
            model.update_weights_momentum(learning_rate, momentum_gamma) if use_momentum else model.update_weights(learning_rate)

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                val_logits = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, val_logits)
                val_loss[global_step] = _val_loss
                # Logits for
                train_logits = model.forward(X_train)
                _train_loss = cross_entropy_loss(Y_train, train_logits)
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
        # Track training loss continuously
        if use_shuffle:
            shuffle_equal(X_train, Y_train)
        logits = model.forward(X_train)

    return model, train_loss, val_loss, train_accuracy, val_accuracy

def shuffle_equal(X:np.ndarray, Y:np.ndarray):
    state_of_random = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state_of_random)
    np.random.shuffle(Y)



if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)
    Xt_mean = np.mean(X_train)
    Xt_std = np.std(X_train)

    X_train = pre_process_images(X_train, Xt_mean, Xt_std)
    X_val = pre_process_images(X_val, Xt_mean, Xt_std)
    X_test = pre_process_images(X_test, Xt_mean, Xt_std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test = one_hot_encode(Y_test, 10)

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    name_trick_used = ["Shuffle", "ImpSigmoid", "ImpWeightInit", "Momentum"]

    # Settings for task 3. Keep all to false for task 2.
    # Adjust range if running all the tricks is not desired.
    first_iteration_done = False
    for i in range(0,5):

        use_shuffle = (i >= 1)
        use_improved_sigmoid = ( i >= 2)
        use_improved_weight_init = ( i >= 3)
        use_momentum = ( i >= 4)


        # lowering the learning rate when using momentum
        if(use_momentum): learning_rate = 0.02

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        model, train_loss, val_loss, train_accuracy, val_accuracy = train(
            model,
            [X_train, Y_train, X_val, Y_val, X_test, Y_test],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_shuffle=use_shuffle,
            use_momentum=use_momentum,
            momentum_gamma=momentum_gamma)

        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Test Cross Entropy Loss:",
              cross_entropy_loss(Y_test, model.forward(X_test)))

        print("Final Train accuracy:",
             calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
              calculate_accuracy(X_val, Y_val, model))
        print("Final Test accuracy:",
              calculate_accuracy(X_test, Y_test, model))

        if(first_iteration_done):
            # Plot loss
            plt.figure(figsize=(10, 8))
            # plt.subplot(1, 2, 1)
            plt.ylim([0.0, .5])
            utils.plot_loss(last_train_loss, "Training Loss")
            utils.plot_loss(last_val_loss, "Validation Loss")
            utils.plot_loss(train_loss, "Training Loss with " + name_trick_used[i-1])
            utils.plot_loss(val_loss, "Validation Loss with " + name_trick_used[i-1])
            plt.xlabel("Number of gradient steps")
            plt.ylabel("Cross Entropy Loss")
            plt.legend()
            plt.savefig("softmax_train" + name_trick_used[i-1] + ".png")

        last_train_loss = train_loss.copy()
        last_val_loss = val_loss.copy()
        first_iteration_done = True

    plt.show()
