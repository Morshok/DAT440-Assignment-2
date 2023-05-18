import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [9, 6]

class ResultType(Enum):
    PER_STEP = 1
    PER_EPISODE = 2

def display_results(list_of_Q_tables, result_type, rewards):
    num_subplots = len(list_of_Q_tables)
    fig, axs = plt.subplots(nrows = 1, ncols = num_subplots + 1, squeeze = False)

    for index, Q_Table in enumerate(list_of_Q_tables):
        axs[0, index].imshow(Q_Table) 
        if(num_subplots > 1):
            axs[0, index].set_title('Q-Table {}: '.format(index + 1))
        else:
            axs[0, index].set_title('Q-Table: ')
        axs[0, index].set_ylabel('State')
        axs[0, index].set_xlabel('Action')
        for (j, i), label in np.ndenumerate(Q_Table):
            axs[0, index].text(i, j, np.round(label, 1), ha = 'center', va = 'center')

    match result_type:
        case ResultType.PER_STEP:
            moving_average = get_moving_average(rewards, 500)
            plt.plot(range(len(moving_average)), moving_average)
        case ResultType.PER_EPISODE:
            moving_average = get_moving_average(rewards, 1)
            plt.plot(range(len(moving_average)), moving_average)
        case _:
            pass

    plt.suptitle("Q-Table(s) and Rewards:")

    plt.tight_layout()
    plt.show()

def get_moving_average(data, window_width):
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    return (cumulative_sum[window_width:] - cumulative_sum[:-window_width]) / window_width