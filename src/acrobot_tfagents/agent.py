import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common


class TFAgentsDQNFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, time_step_spec, action_spec, q_network):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.cfg['training']['learning_rate']
        )
        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_network,
            optimizer=optimizer,
            epsilon_greedy=self.cfg['policy']['epsilon_greedy'],
            target_update_tau=self.cfg['training']['target_update_tau'],
            target_update_period=self.cfg['training']['target_update_period'],
            gamma=self.cfg['training']['gamma'],
            td_errors_loss_fn=common.element_wise_squared_loss,
            use_double_dqn=self.cfg['variants']['double_dqn'],
        )
        agent.initialize()
        return agent
