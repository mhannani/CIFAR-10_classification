from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import os
import pickle


class CifarModel:
    """
    The Cifar helper class to train the model.
    """
    def __init__(self, input_shape, dense_layers, z_shape):
        """
        The class constructor.
        """
        self.name = 'model'
        self.input_shape = input_shape
        self.dense_layers = dense_layers
        self.n_dense_layer = len(dense_layers)
        self.z_shape = z_shape

        # build the architecture of the model
        self._build()

    def _build(self):
        """
        Build the model.
        :return: None
        """

        model_input = Input(shape=self.input_shape, name='model_input')
        x = Flatten(name='flatten_layer')(model_input)

        for i in range(self.n_dense_layer):
            x = Dense(units=i, activation='relu', name=f'{i}_dense_layer')(x)

        model_output = Dense(units=self.z_shape, activation='softmax', name='output')(x)

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
        plot_model(self.model, to_file=os.path.join('models_info', location, 'viz/model.png'))

    def save(self, location):
        """
        Save the model alongside with its representation.
        :param location: String
            The absolute path where to store the model summary
        :return: None
        """

        # Check if the model directory exists
        if not os.path.exists(os.path.join('models_info', location)):
            os.makedirs(os.path.join('models_info', location))
            os.makedirs(os.path.join('models_info', location, 'viz'))
            os.makedirs(os.path.join('models_info', location, 'weights'))

        # Store the model's parameters
        with open(os.path.join('models_info', location, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_shape,
                self.n_dense_layer,
                self.z_shape
            ], f)

        # Plot the model
        self._plot_model(location)
