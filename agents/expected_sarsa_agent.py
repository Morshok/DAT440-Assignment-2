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
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        if(random_init):
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    if(self.terminal_states == None or state not in self.terminal_states): 
                        self.Q[state, action] = self.env.action_space.sample()

    def observe(self, state1, state2, action1, reward, done):
        self.accumulated_rewards += reward

        if(done):
            self.Q[state1, action1] += self.alpha * reward
        else:
            expected_value = np.mean(self.Q[state2, :])
            self.Q[state1, action1] += self.alpha * (reward + self.gamma * expected_value - self.Q[state1, action1])

    def act(self, observation):
        action = 0

        if(np.random.uniform(0, 1) < self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[observation, :])

        return action
    
    def get_accumulated_rewards(self):
        return self.accumulated_rewards
    
    def get_Q_table(self):
        return self.Q