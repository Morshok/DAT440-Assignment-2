import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 0.05
        self.gamma = 0.95
        self.alpha = 5/1000
        self.Q = np.zeros(shape = (state_space.n, action_space.n))

        for state in range(state_space.n):
            for action in range(action_space.n):
                self.Q[state, action] = self.state_space.sample()

        self.current_state = 0
        self.current_action = 0

    def observe(self, observation, reward, done):
        next_state = observation

        self.Q[self.current_state, self.current_action] += self.alpha * (reward + self.gamma * np.argmax(self.Q[next_state, self.current_action]) - self.Q[self.current_state, self.current_action])

        if(done):
            return;

        self.current_state = next_state

    def act(self, observation):
        action = 0
        if(np.random.uniform(0, 1) < self.epsilon):
            action = self.action_space.sample() 
        else:
            action = np.argmax(self.Q[observation,:])
        
        self.current_action = action
        return action
    
    def get_Q_tables(self):
        return [self.Q]
