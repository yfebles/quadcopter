from keras import layers, models, optimizers, initializers
from keras import backend as keras_backend


class DoubleFullyConnctedlayersCriticModel:
    """Critic Model."""

    def __init__(self, state_size, action_size, learning_rate):

        self.model = None
        self.actions = None
        self.Q_values = None
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.model = self._build_model()

        optimizer = optimizers.Adam(self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = keras_backend.gradients(self.Q_values, self.actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = keras_backend.function(inputs=[*self.model.input, keras_backend.learning_phase()], outputs=action_gradients)

    def _build_model(self):

        states = layers.Input(shape=(self.state_size,), name='states')

        layer_x = layers.Dense(units=64, activation=None)(states)
        layer_x = layers.Dense(units=64, activation=None)(layer_x)
        layer_x = layers.Dense(units=64, activation=None)(layer_x)
        layer_x = layers.normalization.BatchNormalization()(layer_x)

        layer_x = layers.Dense(units=64, activation=None)(layer_x)
        layer_x = layers.Dense(units=64, activation=None)(layer_x)
        layer_x = layers.Dense(units=64, activation=None)(layer_x)
        layer_x = layers.normalization.BatchNormalization()(layer_x)

        state_net = layers.Activation('relu')(layer_x)

        self.actions = layers.Input(shape=(self.action_size,), name='actions')
        action_x = layers.Dense(units=64, activation=None)(self.actions)
        action_x = layers.normalization.BatchNormalization()(action_x)
        actions_net = layers.Activation('relu')(action_x)

        # Combine state and action pathways
        network_model = layers.Add()([state_net, actions_net])
        network_model = layers.Activation('relu')(network_model)

        self.Q_values = layers.Dense(units=1, name='q_values')(network_model)

        return models.Model(inputs=[states, self.actions], outputs=self.Q_values)
