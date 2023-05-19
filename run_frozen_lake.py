import plot
import argparse
import numpy as np
import gymnasium as gym
from agents import q_learning_agent, double_q_learning_agent, sarsa_agent, expected_sarsa_agent

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

FROZENLAKE_MAPS = {
    "4x4": [
        "SFFF", 
        "FHFH", 
        "FFFH", 
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

def get_terminal_states(frozen_lake_map):
    terminal_states = []

    num_rows = num_cols = len(frozen_lake_map)
    for row in range(num_rows):
        current_row = frozen_lake_map[row]
        for col in range(num_cols):
            if(current_row[col] == "H" or current_row[col] == "G"):
                terminal_states.append(row * num_rows + col)

    return terminal_states

try:
    env = gym.make(args.env, desc=FROZENLAKE_MAPS["4x4"])
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)

# Experiment Parameters
num_runs = 5
max_num_episodes = 2500

# Reward Lists
q_learning_rewards = []
double_q_learning_rewards = []
sarsa_rewards = []
expected_sarsa_rewards = []

# Agent Parameters
alpha = 0.6
gamma = 0.95
epsilon = 0.05
random_init = False

def get_best_policy(q_tables, best_run_index):
    best_policy = []

    for state in q_tables[best_run_index]:
        best_policy.append(np.argmax(state))
    best_policy = np.array(best_policy).reshape(4, 4)
    
    return best_policy

def run_q_learning_experiment():
    print('Running Q-Learning Experiment...')

    q_tables = []
    best_run_index = 0
    best_run_rewards = 0

    for run in range(num_runs):
        agent = q_learning_agent.Agent(alpha, gamma, epsilon, random_init, env, terminal_states=get_terminal_states(FROZENLAKE_MAPS["4x4"]))
        run_rewards = []
        for episode in range(max_num_episodes):
            state1, _ = env.reset()
            while True:
                action = agent.act(state1)
                state2, reward, done, _, _ = env.step(action)
                agent.observe(state1, state2, action, reward, done)

                state1 = state2

                if(done):
                    run_rewards.append(agent.get_accumulated_rewards())
                    break
        env.close()

        q_learning_rewards.append(run_rewards)
        q_tables.append(agent.get_Q_table())

        current_run_reward = sum(run_rewards)
        if(current_run_reward > best_run_rewards):
            best_run_index = run
            best_run_rewards = current_run_reward

    print('Finished Q-Learning Experiment...')

    return get_best_policy(q_tables, best_run_index)

def run_double_q_learning_experiment():
    print('Running Double Q-Learning Experiment...')

    q_tables = []
    best_run_index = 0
    best_run_rewards = 0

    for run in range(num_runs):
        run_rewards = []
        agent = double_q_learning_agent.Agent(alpha, gamma, epsilon, random_init, env, terminal_states=get_terminal_states(FROZENLAKE_MAPS["4x4"]))

        for episode in range(max_num_episodes):
            state1, _ = env.reset()
            while True:
                action = agent.act(state1)
                state2, reward, done, _, _ = env.step(action)
                agent.observe(state1, state2, action, reward, done)

                state1 = state2

                if(done):
                    run_rewards.append(agent.get_accumulated_rewards())
                    break
        env.close()

        double_q_learning_rewards.append(run_rewards)
        q_tables.append(agent.get_Q_table())

        current_run_reward = sum(run_rewards)
        if(current_run_reward > best_run_rewards):
            best_run_index = run
            best_run_rewards = current_run_reward

    print('Finished Double Q-Learning Experiment...')

    return get_best_policy(q_tables, best_run_index)

def run_sarsa_experiment():
    print('Running SARSA Experiment...')

    q_tables = []
    best_run_index = 0
    best_run_rewards = 0

    for run in range(num_runs):
        run_rewards = []
        agent = sarsa_agent.Agent(alpha, gamma, epsilon, random_init, env, terminal_states=get_terminal_states(FROZENLAKE_MAPS["4x4"]))

        for episode in range(max_num_episodes):
            state1, _ = env.reset()
            action1 = agent.act(state1)
            while True:
                state2, reward, done, _, _ = env.step(action1)
                action2 = agent.act(state2)
                agent.observe(state1, state2, action1, action2, reward, done)
                
                state1 = state2
                action1 = action2

                if(done):
                    run_rewards.append(agent.get_accumulated_rewards())
                    break
        env.close()

        sarsa_rewards.append(run_rewards)
        q_tables.append(agent.get_Q_table())

        current_run_reward = sum(run_rewards)
        if(current_run_reward > best_run_rewards):
            best_run_index = run
            best_run_rewards = current_run_reward

    print('Finished SARSA Experiment...')

    return get_best_policy(q_tables, best_run_index)

def run_expected_sarsa_experiment():
    print('Running Expected SARSA Experiment...')

    q_tables = []
    best_run_index = 0
    best_run_rewards = 0

    for run in range(num_runs):
        run_rewards = []
        agent = expected_sarsa_agent.Agent(alpha, gamma, epsilon, random_init, env, terminal_states=get_terminal_states(FROZENLAKE_MAPS["4x4"]))

        for episode in range(max_num_episodes):
            state1, _ = env.reset()
            action1 = agent.act(state1)
            while True:
                state2, reward, done, _, _ = env.step(action1)
                action2 = agent.act(state2)
                agent.observe(state1, state2, action1, reward, done)
                
                state1 = state2
                action1 = action2

                if(done):
                    run_rewards.append(agent.get_accumulated_rewards())
                    break
        env.close()

        expected_sarsa_rewards.append(run_rewards)
        q_tables.append(agent.get_Q_table())

        current_run_reward = sum(run_rewards)
        if(current_run_reward > best_run_rewards):
            best_run_index = run
            best_run_rewards = current_run_reward

    print('Finished Expected SARSA Experiment...')

    return get_best_policy(q_tables, best_run_index)

best_q_learning_policy = run_q_learning_experiment()
best_double_q_learning_policy = run_double_q_learning_experiment()
best_sarsa_policy = run_sarsa_experiment()
best_expected_sarsa_policy = run_expected_sarsa_experiment()

results_tuple_list = [
    ['Q-Learning', q_learning_rewards, best_q_learning_policy],
    ['Double Q-Learning', double_q_learning_rewards, best_double_q_learning_policy],
    ['SARSA', sarsa_rewards, best_sarsa_policy],
    ['Expected SARSA', expected_sarsa_rewards, best_expected_sarsa_policy]
]

plot.plot_frozen_lake_results(results_tuple_list, max_num_episodes)