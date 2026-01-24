from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam


def build_agent(model, nb_actions, cfg):
    memory = SequentialMemory(limit=cfg['memory_limit'], window_length=cfg['window_length'])
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=cfg['policy']['eps_max'],
        value_min=cfg['policy']['eps_min'],
        value_test=cfg['policy']['eps_test'],
        nb_steps=cfg['policy']['anneal_steps'],
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=cfg['training']['warmup_steps'],
        target_model_update=cfg['training']['target_model_update'],
        gamma=cfg['training']['gamma'],
        policy=policy,
        enable_double_dqn=cfg['variants']['double_dqn'],
        enable_dueling_network=cfg['variants']['dueling_dqn'],
        dueling_type=cfg['variants']['dueling_type'],
    )
    dqn.compile(Adam(learning_rate=cfg['training']['learning_rate']), metrics=['mae'])
    return dqn
