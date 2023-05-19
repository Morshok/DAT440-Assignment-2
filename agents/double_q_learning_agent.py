import numpy as np

class Agent(object):
    def __init__(self, alpha, gamma, epsilon, random_init, env, terminal_states=None):
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon

        self.env = env
        self.terminal_states = terminal_states

        self.accumulated_rewards = 0

        self.__init_Q_table(random_init)

    def __init_Q_table(self, random_init):
        self.Q1 = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.Q2 = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        if(random_init):
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    if(self.terminal_states == None or state not in self.terminal_states): 
                        self.Q1[state, action] = self.env.action_space.sample()
                        self.Q2[state, action] = self.env.action_space.sample()


    def observe(self, state1, state2, action, reward, done):
        self.accumulated_rewards += reward

        if(done):
            if(np.random.uniform(0, 1) < 0.5):
                self.Q1[state1, action] += self.alpha * reward
            else:
                self.Q2[state1, action] += self.alpha * reward
        else:
            if(np.random.uniform(0, 1) < 0.5):
                self.Q1[state1, action] += self.alpha * (reward + self.gamma * np.argmax(self.Q2[state2, :]) - self.Q1[state1, action])
            else:
                self.Q2[state1, action] += self.alpha * (reward + self.gamma * np.argmax(self.Q1[state2, :]) - self.Q2[state1, action])

    def act(self, observation):
        Q = ((self.Q1 + self.Q2) / 2)   # The average of both Q tables

        action = 0

        if(np.random.uniform(0, 1) < self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(Q[observation, :])

        return action
    
    def get_accumulated_rewards(self):
        return self.accumulated_rewards
    
    def get_Q_table(self):
        Q = ((self.Q1 + self.Q2) / 2)   # The average of both Q tables
        return Q 