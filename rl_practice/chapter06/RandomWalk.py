from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

RUNS            = 100

STATE_LABEL     = ['A', 'B', 'C', 'D', 'E']
STATE           = [0,1,2,3,4]

TRUE_VALUES     = [1/6, 2/6, 3/6, 4/6, 5/6]

INIT_VALUE      = 0.5
INIT_STATE      = 2
MOST_LEFT       = STATE[0]-1
MOST_RIGHT      = STATE[-1]+1
ACTION_LEFT     = 0
ACTION_RIGHT    = 1

# Constants
PROBABILITY     = 0.5
DISCOUNT        = 1
ALPHA           = 0.1

def calcRMS(true, observed):
    error = np.array(true - observed)
    error2 = np.square(error)
    return np.sqrt(np.sum(error2)/len(error2))

def TD(nEpisode, alpha=0.1):
    Value = INIT_VALUE*np.ones(len(STATE))
    RMS = np.zeros(nEpisode)
    # get all episodes
    for i in range(nEpisode):
        state = INIT_STATE
        while(True):
            # Take Action and Observe State
            if np.random.binomial(1, PROBABILITY) == ACTION_LEFT:
                next_state = state-1
            else:
                next_state = state+1

            # Observe Reward
            if next_state==MOST_RIGHT:
                reward = 1
            else:
                reward = 0

            if next_state==MOST_LEFT or next_state==MOST_RIGHT:
                Value[state] += alpha*(reward - Value[state])
                break
            else:
                Value[state] += alpha*(reward + DISCOUNT*Value[next_state] - Value[state])
            state = next_state

        a = np.array(TRUE_VALUES)
        b = np.array(Value)
        RMS[i] = calcRMS(a,b)

    return Value, RMS


def RandomWalk():
    states = []
    state = INIT_STATE
    states.append(state)
    while(True):
        # Take Action and Observe State
        if np.random.binomial(1, PROBABILITY) == ACTION_LEFT:
            next_state = state-1
        else:
            next_state = state+1

        states.append(state)
        # Observe Reward
        if next_state==MOST_RIGHT:
            reward = 1
            break
        elif next_state==MOST_LEFT:
            reward = 0
            break
        else:
            reward = 0
        state = next_state

    return states, reward

def MC(nEpisode, alpha=0.1):
    Value = INIT_VALUE*np.ones(len(STATE))
    RMS = np.zeros(nEpisode)
    # get all episodes
    for i in range(nEpisode):
        states, reward = RandomWalk()
        for s in states:
            Value[s] += alpha*(reward - Value[s])

        a = np.array(TRUE_VALUES)
        b = np.array(Value)
        RMS[i] = calcRMS(a,b)

    return Value, RMS

def EstimatedValue():
    nEpisodes = [0, 1, 10, 100]
    Values = []
    for nEpisode in nEpisodes:
        Value, _ = TD(nEpisode)
        Values.append(Value)

    labels = ['0', '1', '10', '100']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(nEpisodes)):
        ax.plot(STATE_LABEL, Values[i], label=labels[i])
    ax.plot(STATE_LABEL, TRUE_VALUES, label='True Value')
    ax.set_xlabel('State')
    ax.set_ylabel('Value')
    plt.grid(which='major', color='black', linestyle='--')
    plt.legend()

def RMSerror():
    nEpisodes = 100
    runs = 100
    alphas = [0.05, 0.1, 0.15]
    alphas2 = [0.01, 0.02, 0.03, 0.04]
    labels = np.array(alphas, dtype=str)
    labels2 = np.array(alphas2, dtype=str)
    RMSs = []
    RMSs2 = []
    for i, alpha in enumerate(alphas):
        RMS = np.zeros(nEpisodes)
        for i in range(runs):
            _, rms = TD(nEpisodes, alpha)
            RMS += rms
        RMSs.append(RMS/nEpisodes)

    for i, alpha in enumerate(alphas2):
        RMS = np.zeros(nEpisodes)
        for i in range(runs):
            _, rms = MC(nEpisodes, alpha)
            RMS += rms
        RMSs2.append(RMS/nEpisodes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(alphas)):
        ax.plot(range(nEpisodes), RMSs[i], label=labels[i])
    for i in range(len(alphas2)):
        ax.plot(range(nEpisodes), RMSs2[i], label=labels2[i])
    ax.set_xlabel('Episodes')
    ax.set_ylabel('RMS Error')
    plt.grid(which='major', color='black', linestyle='--')
    plt.legend()

def main():
    EstimatedValue()
    RMSerror()
    plt.show()

if __name__ == '__main__':
    main()
