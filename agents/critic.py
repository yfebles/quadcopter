from keras import layers, models, optimizers, initializers
from keras import backend as keras_backend

class Critic:
    """Critic Model."""

    def __init__(self, state_size, action_size, learning_rate):

        self.model = None
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Initialize any other variables here

        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        layer1 = layers.Dense(units=32, activation=None)(states)
        layer2 = layers.normalization.BatchNormalization()(layer1)
        layer3 = layers.Activation('relu')(layer2)
        layer4 = layers.Dense(units=64, activation=None)(layer3)
        layer5 = layers.normalization.BatchNormalization()(layer4)
        state_net = layers.Activation('relu')(layer5)

        # Add hidden layer(s) for action pathway
        action1 = layers.Dense(units=64, activation=None)(actions)
        action2 = layers.normalization.BatchNormalization()(action1)
        actions_net = layers.Activation('relu')(action2)

        # Combine state and action pathways
        net = layers.Add()([state_net, actions_net])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = keras_backend.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = keras_backend.function(inputs=[*self.model.input, keras_backend.learning_phase()], outputs=action_gradients)

