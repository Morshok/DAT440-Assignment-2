import argparse
import numpy as np
import pandas as pd
import importlib.util
import gymnasium as gym
import plot_results as pr
import matplotlib.pyplot as plt
from plot_results import ResultType

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', 'agents\\' + args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

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

rewards_per_step = []
rewards_per_episode = []
num_episodes = 5
max_steps = 500
alpha = 0.1
gamma = 0.95
epsilon = 0.05
random_init = True

match args.env:
    case "riverswim:RiverSwim":
        agent = agentfile.Agent(alpha, gamma, epsilon, random_init, env, None)
    case _:
        agent = agentfile.Agent(alpha, gamma, epsilon, random_init, env, get_terminal_states(FROZENLAKE_MAPS["4x4"]))

def run_sarsa_experiment():
    for episode in range(num_episodes):
        state1, _ = env.reset()
        action1 = agent.act(state1)
        truncated = False
        episode_reward = 0

        if(args.env == "riverswim:RiverSwim"):
            step = 0

        while not truncated:
            if(args.env == "riverswim:RiverSwim"):
                step += 1
            else:
                env.render()

            state2, reward, done, truncated, _ = env.step(action1)
            rewards_per_step.append(reward)
            episode_reward += reward
            action2 = agent.act(state2)
            agent.observe(state1, state2, action1, action2, reward)

            state1 = state2
            action1 = action2

            if(args.env == "riverswim:RiverSwim"):
                if(step >= max_steps):
                    break
            if(done or truncated):
                break
        rewards_per_episode.append(episode_reward)
    if(args.env == "riverswim:RiverSwim"):
        pr.display_results(agent.get_Q_table(), ResultType.PER_STEP, rewards_per_step)
    else:
        pr.display_results(agent.get_Q_table(), ResultType.PER_EPISODE, rewards_per_episode)
    env.close()

def run_q_learning_experiment():
    for episode in range(num_episodes):
        state1, _ = env.reset()
        truncated = False
        episode_reward = 0

        if(args.env == "riverswim:RiverSwim"):
            step = 0

        while not truncated:

            if(args.env == "riverswim:RiverSwim"):
                step += 1
            else:
                env.render()
            
            action = agent.act(state1)
            state2, reward, done, truncated, _ = env.step(action)
            rewards_per_step.append(reward)
            episode_reward += reward
            agent.observe(state1, state2, action, reward)

            state1 = state2

            if(args.env == "riverswim:RiverSwim"):
                if(step >= max_steps):
                    break
            if(done or truncated):
                break
        rewards_per_episode.append(episode_reward)
    if(args.env == "riverswim:RiverSwim"):
        pr.display_results(agent.get_Q_table(), ResultType.PER_STEP, rewards_per_step)
    else:
        pr.display_results(agent.get_Q_table(), ResultType.PER_EPISODE, rewards_per_episode)
    env.close()

match args.agentfile:
    case "sarsa_agent.py" | "expected_sarsa_agent.py":
        run_sarsa_experiment()
    case "q_learning_agent.py" | "double_q_learning_agent.py":
        run_q_learning_experiment()
    case _:
        pass