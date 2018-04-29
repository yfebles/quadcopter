from keras import layers, models, optimizers, initializers
from keras import backend as keras_backend


class FullyConnectedActorModel:
    """"""

    def __init__(self, state_size, action_size, action_low, action_high, learning_rate):

        # init fields
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.learning_rate = learning_rate
        self.action_range = self.action_high - self.action_low

        self.model, actions = self._build_model()

        # Loss function. Action-value
        gradients = layers.Input(shape=(self.action_size,))
        loss = keras_backend.mean(-gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = keras_backend.function(inputs=[self.model.input, gradients, keras_backend.learning_phase()],
                                               outputs=[], updates=updates_op)

    def _build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')

        # create all layers for model. first an up to 128 node layers start
        layer_x = layers.Dense(units=32, activation=None)(states)
        layer_x = layers.Dense(units=64, activation=None)(layer_x)

        # fully connected 3 layers sub system
        layer_x = layers.Dense(units=128, activation=None)(layer_x)
        layer_x = layers.Dense(units=128, activation=None)(layer_x)
        layer_x = layers.Dense(units=128, activation=None)(layer_x)

        # down sample to output subsytem
        layer_x = layers.Dense(units=64, activation=None)(layer_x)
        layer_x = layers.Dense(units=32, activation=None)(layer_x)

        # output layer
        layer_x = layers.Activation('relu')(layer_x)

        # Add final output layer with sigmoid activation
        w_init = initializers.RandomUniform(minval=-0.001, maxval=0.001)
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions', kernel_initializer=w_init)(layer_x)

        # scale the action output by dimensions
        scaled_actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
        return models.Model(inputs=states, outputs=scaled_actions), scaled_actions