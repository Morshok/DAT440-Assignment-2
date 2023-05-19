import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

class FrozenLakeDirection(Enum):
    Left = 0
    Down = 1
    Right = 2
    Up = 3

class RiverSwimDirection(Enum):
    Backward = 0
    Forward = 1

def plot_frozen_lake_results(results_list, num_episodes):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 4)
    reward_results_plot = fig.add_subplot(gs[0:2, 0:2])
    reward_results_plot.set_ylabel('Moving Average Reward')
    reward_results_plot.set_xlabel('Episode')
    reward_results_plot.set_box_aspect(1)

    policy_plots = [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3])
    ]

    for index, (algorithm, rewards, best_policy) in enumerate(results_list):
        random_color = __generate_random_color()
        reward_plot = pd.DataFrame(rewards)
        reward_results_plot.plot(
            reward_plot.mean(axis = 0), 
            color = random_color, 
            label=algorithm
        )
        reward_results_plot.fill_between(
            range(num_episodes), 
            (reward_plot.mean(axis = 0) - reward_plot.var() * 0.05), 
            (reward_plot.mean(axis = 0) + reward_plot.var() * 0.05), 
            color = random_color,
            alpha=0.1
        )
        policy_plot = policy_plots[index]
        policy_plot.imshow(best_policy)
        for (j, i), label in np.ndenumerate(best_policy):
            policy_plot.text(i, j, FrozenLakeDirection(np.round(label, 1)).name, ha = 'center', va = 'center')
        policy_plot.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        policy_plot.set_title('Best ' + algorithm + ' Policy: ')
        policy_plot.set_box_aspect(1)
    reward_results_plot.legend(loc='upper left')

    # Display plot in fullscreen mode
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

def plot_riverswim_results(results_list, num_steps):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 4)
    reward_results_plot = fig.add_subplot(gs[0:2, 0:2])
    reward_results_plot.set_title('Moving Average Rewards over 5 Runs (RiverSwim): ')
    reward_results_plot.set_ylabel('Moving Average Reward')
    reward_results_plot.set_xlabel('Step')
    reward_results_plot.set_box_aspect(1)

    policy_plots = [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3])
    ]

    for index, (algorithm, rewards, best_policy) in enumerate(results_list):
        random_color = __generate_random_color()
        reward_plot = pd.DataFrame(rewards)

        reward_results_plot.plot(
            reward_plot.mean(axis = 0), 
            color = random_color, 
            label=algorithm
        )
        reward_results_plot.fill_between(
            range(num_steps), 
            (reward_plot.mean(axis = 0) - reward_plot.var() * 0.05), 
            (reward_plot.mean(axis = 0) + reward_plot.var() * 0.05), 
            color = random_color,
            alpha=0.1
        )
        policy_plot = policy_plots[index]
        policy_plot.imshow(best_policy)
        for (j, i), label in np.ndenumerate(best_policy):
            policy_plot.text(i, j, RiverSwimDirection(np.round(label, 1)).name, color='red', ha = 'center', va = 'center')
        policy_plot.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        policy_plot.set_title('Best ' + algorithm + ' Policy: ')
        policy_plot.set_box_aspect(1)
    reward_results_plot.legend(loc='upper left')

    # Display plot in fullscreen mode
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

def __generate_random_color():
    red = np.random.uniform(0, 1)
    green = np.random.uniform(0, 1)
    blue = np.random.uniform(0, 1)

    return (red, green, blue)