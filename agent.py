import random
import collections
import numpy as np
import tensorflow as tf
from tensorboard import TensorflowLogger
from keras import models, layers, optimizers

class UnoAgent:

    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 512
    DISCOUNT_FACTOR = 0.7
    MODEL_UPDATE_FREQUENCY = 20

    def __init__(self, state_size, action_count):
        print('Initializing agent...')
        self.initialized = False
        self.logger = TensorflowLogger('logs')

        # initialize the prediction model and a clone of it, the target model
        self.model = self.create_model(state_size, action_count)
        self.target_model = self.create_model(state_size, action_count)
        self.target_model.set_weights(self.model.get_weights())

        # initialize the replay memory
        self.replay_memory = collections.deque(maxlen=self.REPLAY_MEMORY_SIZE)

    def create_model(self, input_size, output_size):
        # define the model architecture
        model = models.Sequential()
        model.add(layers.Dense(units=64, activation='relu', input_shape=(input_size,)))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=output_size, activation='linear'))

        # compile the model
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        # add a state transition to the replay memory
        self.replay_memory.append(transition)

    def predict(self, state):
        # return the index of the action with the highest predicted Q value
        return np.argmax(self.model.predict(np.array(state).reshape(-1, *state.shape))[0])

    def train(self):
        counter = 0
        while True:
            if len(self.replay_memory) < self.BATCH_SIZE:
                # wait until enough data is collected
                continue

            # get minibatch from replay memory
            minibatch = np.array(random.sample(self.replay_memory, self.BATCH_SIZE))

            # get states from minibatch
            states = np.array(list(minibatch[:,0]))
            # predict Q values for all states in the minibatch
            q_values = self.model.predict(states)
            # estimate the maximum future reward
            max_future_q = np.max(self.model.predict(np.array(list(minibatch[:,3]))), axis=1)

            for i in range(len(minibatch)):
                action, reward, done = minibatch[i,1], minibatch[i,2], minibatch[i,4]

                # update the Q value of the chosen action
                q_values[i,action] = reward
                if not done:
                    # add the discounted maximum future reward if the current transition was not the last in an episode
                    q_values[i,action] += self.DISCOUNT_FACTOR * max_future_q[i]

            # train the model on the minibatch
            hist = self.target_model.fit(x=states, y=q_values, batch_size=self.BATCH_SIZE, verbose=0)
            self.logger.scalar('loss', hist.history['loss'][0])
            self.logger.scalar('acc', hist.history['acc'][0])
            self.logger.flush()

            counter += 1
            if counter == self.MODEL_UPDATE_FREQUENCY:
                # update the prediction model
                self.model.set_weights(self.target_model.get_weights())
                if not self.initialized:
                    print('Agent initialized')
                    self.initialized = True
                counter = 0
