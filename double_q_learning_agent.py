import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05
        self.gamma = 0.95
        self.alpha = 5/1000
        self.Q1 = np.zeros(shape = (state_space.n, action_space.n))
        self.Q2 = np.zeros(shape = (state_space.n, action_space.n))

        for state in range(state_space.n):
            for action in range(action_space.n):
                self.Q1[state, action] = self.state_space.sample()
                self.Q2[state, action] = self.state_space.sample()

        self.current_state = 0
        self.current_action = 0

    def observe(self, observation, reward, done):
        next_state = observation

        if(np.random.uniform(0, 1) < 0.5):
            self.Q1[self.current_state, self.current_action] += self.alpha * (reward + self.gamma * np.argmax(self.Q2[next_state, self.current_action]) - self.Q1[self.current_state, self.current_action])
        else:
            self.Q2[self.current_state, self.current_action] += self.alpha * (reward + self.gamma * np.argmax(self.Q1[next_state, self.current_action]) - self.Q2[self.current_state, self.current_action])

        if(done):
            return;

        self.current_state = next_state

    def act(self, observation):
        Q = np.concatenate((self.Q1, self.Q2), axis = 0)

        if(np.random.uniform(0, 1) < self.epsilon):
            self.current_action = self.action_space.sample() 
        else:
            self.current_action = np.argmax(Q[observation,:])
        
        return self.current_action

    def get_Q_tables(self):
        return [self.Q1, self.Q2]
