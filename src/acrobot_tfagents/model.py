from tf_agents.networks import q_network
from tf_agents.networks import dueling_q_network


class TFAgentsNetworkBuilder:
    def __init__(self, fc_layer_params=(128, 128), network_type='q'):
        self.fc_layer_params = tuple(fc_layer_params)
        self.network_type = network_type

    def build(self, observation_spec, action_spec):
        if self.network_type == 'dueling':
            return dueling_q_network.DuelingQNetwork(
                observation_spec,
                action_spec,
                fc_layer_params=self.fc_layer_params,
            )
        return q_network.QNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=self.fc_layer_params,
        )
