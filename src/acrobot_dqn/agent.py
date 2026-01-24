from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam


class DQNAgentFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, model, nb_actions):
        memory = SequentialMemory(
            limit=self.cfg['memory_limit'],
            window_length=self.cfg['window_length'],
        )
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr='eps',
            value_max=self.cfg['policy']['eps_max'],
            value_min=self.cfg['policy']['eps_min'],
            value_test=self.cfg['policy']['eps_test'],
            nb_steps=self.cfg['policy']['anneal_steps'],
        )

        dqn = DQNAgent(
            model=model,
            nb_actions=nb_actions,
            memory=memory,
            nb_steps_warmup=self.cfg['training']['warmup_steps'],
            target_model_update=self.cfg['training']['target_model_update'],
            gamma=self.cfg['training']['gamma'],
            policy=policy,
            enable_double_dqn=self.cfg['variants']['double_dqn'],
            enable_dueling_network=self.cfg['variants']['dueling_dqn'],
            dueling_type=self.cfg['variants']['dueling_type'],
        )
        dqn.compile(
            Adam(learning_rate=self.cfg['training']['learning_rate']),
            metrics=['mae'],
        )
        return dqn


def build_agent(model, nb_actions, cfg):
    return DQNAgentFactory(cfg).build(model, nb_actions)
