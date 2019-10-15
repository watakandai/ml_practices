from __future__ import print_function
import time
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

RUNS            = 100

STATE_LABEL     = ['A', 'B', 'C', 'D', 'E']
NUM_STATES      = 19
STATE           = np.arange(0, NUM_STATES)
TRUE_VALUES     = np.arange(-20, 22, 2) / 20.0
TRUE_VALUES[0]  = TRUE_VALUES[-1] = 0
TRUE_VALUES     = TRUE_VALUES[1:20]

INIT_VALUE      = 0.5
INIT_STATE      = STATE[int(len(STATE)/2)]
MOST_LEFT       = STATE[0]-1
MOST_RIGHT      = STATE[-1]+1
ACTION_LEFT     = 0
ACTION_RIGHT    = 1

# Constants
PROBABILITY     = 0.5
GAMMA           = 1


def calcRMS(true, observed):
    error = np.array(true - observed)
    error2 = pow(error,2)
    return np.sqrt(np.sum(error2)/len(error2))


def TD(nstep, nEpisode, alpha=0.1):
    RMS = 0.0
    Value = INIT_VALUE*np.ones(len(STATE))
    for i in range(nEpisode):
        state = INIT_STATE
        State = [state]
        Reward = [0]
        T = math.inf
        t = 0
        tau = 0
        while(tau!=T-1):
            if t < T:
                # Take Action and Observe State
                if np.random.binomial(1, PROBABILITY) == ACTION_LEFT:
                    next_state = state-1
                else:
                    next_state = state+1

                # Observe Reward
                if next_state==MOST_RIGHT:
                    reward = 1
                elif next_state==MOST_LEFT:
                    reward = -1
                else:
                    reward = 0

                State.append(next_state)
                Reward.append(reward)

                if next_state==MOST_LEFT or next_state==MOST_RIGHT:
                    T = t+1

            tau = t-nstep+1
            if tau>=0:
                G = 0.0
                for i in range(tau+1, min(tau+nstep, T)+1):
                    G += pow(GAMMA, i-tau-1)*Reward[i]
                if tau+nstep<T:
                    G += pow(GAMMA, nstep)*Value[State[tau+nstep]]
                Value[State[tau]] += alpha*(G - Value[State[tau]])

            state = next_state
            t += 1

        RMS += calcRMS(TRUE_VALUES, Value)

    return Value, RMS/nEpisode


def nStepRandomWalk():
    nEpisodes = 10
    runs = 100
    alphas = np.arange(0, 1.1, 0.1)
    nsteps = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    labels = np.array(nsteps, dtype=str)
    RMSs = []
    Ns = []
    for i, nstep in enumerate(nsteps):
        print('n = {}'.format(nstep))
        RMS = np.zeros(len(alphas))
        for j, alpha in enumerate(alphas):
            rms = 0.0
            for k in range(runs):
                _, r = TD(nstep, nEpisodes, alpha)
                rms += r
            RMS[j] = rms/runs
            print('alpha={}, RMS={}'.format(alpha, RMS[j]))

        RMSs.append(RMS)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(nsteps)):
        ax.plot(alphas, RMSs[i], label=labels[i])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('RMS Error')
    plt.grid(which='major', color='black', linestyle='--')
    plt.legend()

def main():
    nStepRandomWalk()
    plt.show()

if __name__ == '__main__':
    main()
