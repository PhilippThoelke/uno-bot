import time
import threading
import numpy as np
from agent import UnoAgent
from environment import UnoEnvironment

PLAYER_COUNT = 4
COLLECTOR_THREADS = 2
REPORT_FREQUENCY = 1000
EPSILON_DECAY = 0.999995
MIN_EPSILON = 0.01

def run(agent):
    # initialize environment
    epsilon = 1
    env = UnoEnvironment(PLAYER_COUNT)

    counter = 0
    metrics = {'cumulative_reward': [], 'mean_reward': [], 'episode_length': [], 'epsilon': []}
    while True:
        done = False
        state = None

        rewards = []
        # run one episode
        while not done:
            if state is None or np.random.sample() < epsilon:
                # choose a random action
                action = np.random.randint(env.action_count())
            else:
                # choose an action from the policy
                action = agent.predict(state)

            new_state, reward, done = env.step(action)
            rewards.append(reward)

            if state is not None:
                # include the current transition in the replay memory
                agent.update_replay_memory((state, action, reward, new_state, done))
            state = new_state

            if agent.initialized:
                # decay epsilon
                epsilon *= EPSILON_DECAY
                epsilon = max(epsilon, MIN_EPSILON)
        # reset the environment for the next episode
        env.reset()
        metrics['cumulative_reward'].append(np.sum(rewards))
        metrics['mean_reward'].append(np.mean(rewards))
        metrics['episode_length'].append(len(rewards))
        metrics['epsilon'].append(epsilon)

        counter += 1
        if counter % REPORT_FREQUENCY == 0:
            print(f'Metrics {counter}')
            [print(f'{name}: {np.mean(metric)}') for name, metric in metrics.items()]
            print()
            metrics = {'cumulative_reward': [], 'mean_reward': [], 'episode_length': [], 'epsilon': []}

if __name__ == '__main__':
    dummy_env = UnoEnvironment(1)
    agent = UnoAgent(dummy_env.state_size(), dummy_env.action_count())
    del dummy_env

    for _ in range(COLLECTOR_THREADS):
        threading.Thread(target=run, args=(agent,), daemon=True).start()

    agent.train()
    