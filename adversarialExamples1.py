import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class FullyConnectedNN:
    def __init__(self, num_of_layers, data, batch_size=128, epochs=5, num_of_input_nodes=784):
        self.num_of_layers = num_of_layers
        self.num_of_input_nodes = num_of_input_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.data = data
        self.model = None
        # TODO - THIS FIELD IS IN CASE WE WANT TO PLOT ANYTHING. IF NOT - ERASE!!!!!!
        self.history = None
        self._set_model()

    def _set_model(self):
        """
        sets the self.model field, which is the network itself - layers and activation:
        :return: no return value.
        """
        self.model = Sequential()
        # first inner layer gets the input:
        self.model.add(Dense(200, input_dim=self.num_of_input_nodes, activation="relu",
                             use_bias=False, kernel_initializer="glorot_normal"))
        for i in range(self.num_of_layers - 2):
            self.model.add(Dense(200, activation="relu", use_bias=False,
                                 kernel_initializer="glorot_normal"))

        self.model.add(Dense(1, use_bias=False, kernel_initializer="glorot_normal"))

        # compile:
        self.model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

    def fit_model(self):
        self.history = self.model.fit(self.data.x_train, self.data.y_train, batch_size=self.batch_size,
                                      epochs=self.epochs, validation_data=(self.data.x_test, self.data.y_test))

    def predict(self, image):
        return self.model.predict(image)

    def print_accuracy(self):
        print("Base loss and accuracy on regular images:",
              self.model.evaluate(x=self.data.x_test, y=self.data.y_test, verbose=0))


class MnistData:
    num_of_features = 784
    # num_of_digits = 10

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # PREPROCESSING OF THE DATA:
        # 1. Convert to float32
        self.x_train, self.x_test = np.array(self.x_train, np.float32), np.array(self.x_test, np.float32)

        # 2. Flatten images to 1-D vector of 784 features (28*28)
        self.x_train = self.x_train.reshape([-1, MnistData.num_of_features])
        self.x_test = self.x_test.reshape([-1, MnistData.num_of_features])

        # 3. For each row in the x_train and x_test - subtract the mean:
        self.x_train = self.x_train - np.mean(self.x_train, axis=1, keepdims=True)
        self.x_test = self.x_test - np.mean(self.x_test, axis=1, keepdims=True)

        # 4. normalizing the training and test input to have squared norm of 784, in 2 steps:
        self.x_train = self.x_train / np.linalg.norm(self.x_train, ord=2, axis=1, keepdims=True)
        self.x_train = self.x_train * np.sqrt(MnistData.num_of_features)

        self.x_test = self.x_test / np.linalg.norm(self.x_test, ord=2, axis=1, keepdims=True)
        self.x_test = self.x_test * np.sqrt(MnistData.num_of_features)

        # 5. changing labels: if digit is even - the label becomes -1, if digit is odd - the label becomes 1:
        self.y_train = np.array([0 if digit % 2 == 0 else 1 for digit in self.y_train])
        self.y_test = np.array([0 if digit % 2 == 0 else 1 for digit in self.y_test])


class AdversarialGenerator:
    def __init__(self, neural_network):
        self.network = neural_network

    def create_perturbed_image_gd_step(self, image, etta):
        tf_image = tf.cast(image, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(tf_image)
            prediction = self.network.model(tf_image)

        gradient = tape.gradient(prediction, tf_image)
        perturbed_image = image - AdversarialGenerator.pseudo_sign(prediction) * etta * gradient.numpy()
        return perturbed_image, prediction

    @staticmethod
    def pseudo_sign(num):
        if num < 0.5:
            return -1
        else:
            return 1

    def iterate_gd_until_label_change(self, image, etta=0.1):
        original_prediction = self.network.model(image)
        perturbed_image, prediction = self.create_perturbed_image_gd_step(image, etta)
        iteration = 0
        while AdversarialGenerator.same_label(original_prediction, prediction) and iteration < 1000000:
            perturbed_image, prediction = self.create_perturbed_image_gd_step(perturbed_image, etta)
            iteration += 1
        if iteration == 100000:
            print("distance: ", np.linalg.norm(image - perturbed_image))
            print("!!!!!!!!!!!!!!!!!!! DID NOT GO WELL!!!!!!!!!!!!!")

        return perturbed_image, prediction

    @staticmethod
    def same_label(predict1, predict2):
        if predict1 >= 0.5 and predict2 >= 0.5:
            return True
        elif predict1 < 0.5 and predict2 < 0.5:
            return True
        else:
            return False


mnist_data = MnistData()
num_of_iterations = 1000
indices_array = np.random.choice(len(mnist_data.x_test), num_of_iterations)
num_of_layers_array = [2, 4, 8, 16]
# num_of_layers_array = [8]
for layer_num in num_of_layers_array:
    network1 = FullyConnectedNN(layer_num, mnist_data)
    network1.fit_model()
    network1.print_accuracy()
    sum_of_distances = 0
    for index in indices_array:
        adversarial_gen = AdversarialGenerator(network1)
        perturbed_image1, prediction1 = adversarial_gen.iterate_gd_until_label_change(
            mnist_data.x_test[index].reshape(1, MnistData.num_of_features))
        sum_of_distances += np.linalg.norm(mnist_data.x_test[index].reshape(1, MnistData.num_of_features) - perturbed_image1)

        print("distance of current iteration is: ",
              np.linalg.norm(mnist_data.x_test[index].reshape(1, MnistData.num_of_features) - perturbed_image1))

    average = sum_of_distances / num_of_iterations
    print("Average of distances between image and perturbation among " + str(num_of_iterations) +
          " testing images is: ", average)
