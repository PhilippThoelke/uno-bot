import sys
import threading
import numpy as np
from agent import UnoAgent
from environment import UnoEnvironment

PLAYER_COUNT = 4
COLLECTOR_THREADS = 2
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.999999
MIN_EPSILON = 0.01

def run(agent):
    # initialize environment
    epsilon = INITIAL_EPSILON
    env = UnoEnvironment(PLAYER_COUNT)

    counter = 0
    while True:
        done = False
        state = None

        rewards = []
        # run one episode
        while not done:
            if state is None or np.random.sample() < epsilon or not agent.initialized:
                # choose a random action
                action = np.random.randint(env.action_count())
            else:
                # choose an action from the policy
                action = agent.predict(state)

            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if state is not None:
                # include the current transition in the replay memory
                agent.update_replay_memory((state, action, reward, new_state, done))
            state = new_state

            if agent.initialized:
                # decay epsilon
                epsilon *= EPSILON_DECAY
                epsilon = max(epsilon, MIN_EPSILON)

        # log metrics
        agent.logger.scalar('cumulative_reward', np.sum(rewards))
        agent.logger.scalar('mean_reward', np.mean(rewards))
        agent.logger.scalar('game_length', len(rewards))
        agent.logger.scalar('epsilon', epsilon)

        # reset the environment for the next episode
        env.reset()

if __name__ == '__main__':
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # initialize the training agent
    dummy_env = UnoEnvironment(1)
    agent = UnoAgent(dummy_env.state_size(), dummy_env.action_count(), model_path)
    del dummy_env

    # start up threads for experience collection
    for _ in range(COLLECTOR_THREADS):
        threading.Thread(target=run, args=(agent,), daemon=True).start()

    # blocking call to agent, invoking an endless training loop
    agent.train()
