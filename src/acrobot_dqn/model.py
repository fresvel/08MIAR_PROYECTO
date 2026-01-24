from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


class DQNModelBuilder:
    def __init__(self, hidden_units=(128, 128), activation='relu'):
        self.hidden_units = tuple(hidden_units)
        self.activation = activation

    def build(self, input_shape, nb_actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + tuple(input_shape)))
        for units in self.hidden_units:
            model.add(Dense(units, activation=self.activation))
        model.add(Dense(nb_actions, activation='linear'))
        return model


def build_model(input_shape, nb_actions, hidden_units=(128, 128), activation='relu'):
    builder = DQNModelBuilder(hidden_units=hidden_units, activation=activation)
    return builder.build(input_shape, nb_actions)
