from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


def build_model(input_shape, nb_actions, hidden_units=(128, 128), activation='relu'):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + tuple(input_shape)))
    for units in hidden_units:
        model.add(Dense(units, activation=activation))
    model.add(Dense(nb_actions, activation='linear'))
    return model
