"""
Most of the algorithm used here was taken from https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
"""


import numpy as np
from tensorflow import keras
from DeepQAgent.agent_history import AgentHistory
import json


class Agent:
    def __init__(self, model, env, configfile='config.json', render=False, verbose=0):
        configs = json.load(open(configfile))
        # Initialize hyper-parameters
        self._epsilon = configs.get('epsilon') or 1
        self._min_epsilon = configs.get('min_epsilon') or 0.01
        self._max_epsilon = configs.get('max_epsilon') or 1
        self._epsilon_interval = self._max_epsilon - self._min_epsilon
        self._decay = configs.get('decay') or 0.01
        self._learning_rate = configs.get('learning_rate') or 0.7
        self._discount_factor = configs.get('discount_factor') or 0.618
        self._train_episodes = configs.get('train_episodes') or 300
        self._steps_per_train = configs.get('steps_per_train') or 4
        self._steps_per_update = configs.get('steps_per_update') or 100
        self._batch_size = configs.get('batch_size') or 128
        self._min_replay_size = configs.get('min_replay_size') or 1000
        self._max_memory_len = configs.get('max_memory_length') or 10000

        # Initialize counters and other attributes
        self._steps = 0
        self._episode = 0
        self._episode_reward = 0
        self._model = model
        self._target_model = keras.models.clone_model(model)
        self._agent_history = AgentHistory(self._max_memory_len)
        self._env = env
        self._render = render
        self._verbose = verbose

    def _get_training_data(self, observations, new_observations, rewards, actions, done_array):
        # Predict Q-Values for current and future observations
        current_qs_list = self._model.predict(observations)
        future_qs_list = self._target_model.predict(new_observations)
        max_future_qs = np.max(future_qs_list, axis=1)
        # Apply the discount for future rewards where not done
        max_future_qs = rewards + self._discount_factor * max_future_qs * (1 - done_array)
        indexes = np.arange(len(actions)), actions
        # Apply the Bellman Equation
        current_qs_list[indexes] = (1 - self._learning_rate) * current_qs_list[indexes] + self._learning_rate * max_future_qs
        return observations, current_qs_list

    def train_model(self):
        if self._agent_history.length < self._min_replay_size:
            return

        batch = self._agent_history.get_sample(self._batch_size)
        x, y = self._get_training_data(*batch)
        self._model.fit(x, y, batch_size=self._batch_size, verbose=0, shuffle=True)

    def choose_action(self, observation):
        # If random number smaller than epsilon - take random action,
        # otherwise - take the action with max predicted Q-Value
        if np.random.rand() <= self._epsilon:
            action = self._env.action_space.sample()
        else:
            encoded = observation
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            predicted = self._model.predict(encoded_reshaped).flatten()
            action = np.argmax(predicted)
        return action

    def register_new_observation(self, observation, action, new_observation, reward, done):
        self._steps += 1
        # Append buffers
        self._agent_history.update(action, observation, new_observation, done, reward)

        # Train the model if enough steps have passed
        if self._steps % self._steps_per_train == 0 or done:
            self.train_model()

        # Register episode reward
        self._episode_reward += reward

        if done:
            if self._verbose > 1:
                print(f'Episode {self._episode} reward: {self._episode_reward}')
            # Update the target model if enough steps have passed
            if self._steps >= self._steps_per_update:
                if self._verbose > 0:
                    print('Copying main network weights to the target network weights')
                self._target_model.set_weights(self._model.get_weights())
                self._steps = 0
            # Update episode counter and epsilon value
            self._episode += 1
            self._epsilon = self._min_epsilon + (self._max_epsilon - self._min_epsilon) * np.exp(-self._decay * self._episode)

    def train_agent(self):
        # Run the loop of getting observation, choosing action and its impact and updating the network
        while self._episode < self._train_episodes:
            self._episode_reward = 0
            observation = self._env.reset()
            done = False
            while not done:
                if self._render:
                    self._env.render()

                action = self.choose_action(observation)
                new_observation, reward, done, info = self._env.step(action)
                self.register_new_observation(observation, action, new_observation, reward, done)
                observation = new_observation

        self._env.close()

