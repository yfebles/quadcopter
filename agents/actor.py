from keras import layers, models, optimizers, initializers
from keras import backend as keras_backend


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, learning_rate):
        """Initialize parameters and build model.
        """
        self.model = None
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learning_rate = learning_rate

        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation=None)(states)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=64, activation=None)(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=32, activation=None)(net)
        net = layers.normalization.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add final output layer with sigmoid activation
        w_init = initializers.RandomUniform(minval=-0.001, maxval=0.001)
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions', kernel_initializer=w_init)(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = keras_backend.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras_backend.function( inputs=[self.model.input, action_gradients, keras_backend.learning_phase()], outputs=[], updates=updates_op)

