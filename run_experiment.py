import argparse
import gymnasium as gym
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

try:
    env = gym.make(args.env, render_mode="human")
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0", render_mode="human")
    print("Loaded", args.env)

rewards = []
mean_rewards_per_episode = []
state_space = env.observation_space
action_space = env.action_space

agent = agentfile.Agent(state_space, env.action_space)

num_episodes = 5
num_iterations = 10

def run_sarsa_experiment():
    for episode in range(num_episodes):
        observation1, info = env.reset()
        action1 = agent.act(observation1)
        episode_rewards = []
        for iteration in range(num_iterations): 
            env.render()
            observation2, reward, done, truncated, info = env.step(action1)
            action2 = agent.act(observation2)
            rewards.append(reward)
            episode_rewards.append(reward)
            agent.observe(observation2, action2, reward, done)
            
            if done:
                break;
        mean_rewards_per_episode.append(np.mean(episode_rewards))

def run_q_learning_experiment():
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_rewards = []
        for iteration in range(num_iterations): 
            env.render()
            action = agent.act(observation)
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            episode_rewards.append(reward)
            agent.observe(observation, reward, done)
            
            if done:
                observation, info = env.reset()
        mean_rewards_per_episode.append(np.mean(episode_rewards))
    agent.visualize_Q_table()

def plot_rewards(num_episodes, mean_rewards):
    reward_plot = pd.DataFrame(data={'episode': np.arange(1, num_episodes + 1),
                       'mean_rewards': mean_rewards})
    reward_plot.plot(x='episode', y='mean_rewards', title='Mean Rewards Per Episode')
    plt.show()

match args.agentfile:
    case "q_learning_agent.py" | "double_q_learning_agent.py":
        run_q_learning_experiment()
    case "sarsa_agent.py" | "expected_sarsa_agent.py":
        run_sarsa_experiment()
    case _:
        pass
 
plot_rewards(num_episodes, mean_rewards_per_episode)

print("Done!")
env.close()