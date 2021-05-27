from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import os


class CifarModel:
    """
    The Cifar helper class to train the model.
    """
    def __init__(self, input_shape, n_dense_layer, z_shape):
        """
        The class constructor.
        """
        self.name = 'model'
        self.input_shape = input_shape
        self.n_dense_layer = n_dense_layer
        self.z_shape = z_shape

        # build the architecture of the model
        self._build()

    def _build(self):
        """
        Build the model.
        :return: None
        """

        model_input = Input(shape=self.input_shape, name='model_input')
        x = Flatten()(model_input)

        for i in range(self.n_dense_layer):
            x = Dense(units=i, activation='relu')(x)
        model_output = 10


        # Define the model
        self.model = Model(model_input, model_output)

    def compile(self, lr):
        """
        Compile the model with an optimizer and a loss function.
        :param lr: double
            The learning rate.
        :return: None
        """

        self.lr = lr

        # The optimized
        optimizer = Adam(learning_rate=lr)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer, metrics=['accuracy'])

    def _plot_model(self, location):
        """
        Plot the model and save it.
        :param location: String
            The absolute path where to store the model summary
        :return: None
        """
        plot_model(self.model, to_file=os.path.join(location, 'viz/model.png'))


