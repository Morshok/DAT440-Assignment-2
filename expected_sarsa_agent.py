import numpy as np

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
                self.Q[state, action] = self.action_space.sample()

        self.current_state = 0
        self.current_action = 0
        self.actions_taken = 0


    def observe(self, observation, next_action, reward, done):
        expected_value = np.mean(self.Q[observation,:])
        self.Q[self.current_state, self.current_action] += self.alpha * (reward + self.gamma * self.Q[observation, next_action] - self.Q[self.current_state, self.current_action])

        if(done):
            self.Q[self.current_state, self.current_action] += self.alpha * (reward - self.Q[self.current_state, self.current_action])
        else:
            self.Q[self.current_state, self.current_action] += self.alpha * (reward + self.gamma * expected_value - self.Q[self.current_state, self.current_action])

        self.current_state = observation
        self.current_action = next_action

    def act(self, observation):
        action = 0
        if(np.random.uniform(0, 1) < self.epsilon):
            action = self.action_space.sample() 
        else:
            action = np.argmax(self.Q[observation,:])

        self.actions_taken += 1
        if(self.actions_taken % 2 == 0):
            self.actions_taken = 0
        if(self.actions_taken % 2 == 1):
            self.current_action = action
            self.current_state = observation

        return action
    
    def get_Q_tables(self):
        return [self.Q]