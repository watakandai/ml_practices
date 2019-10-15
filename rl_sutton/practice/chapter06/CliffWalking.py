from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#########################################################
# Trajectory = STATE & ACTION                           #
#########################################################
##################Global Variables#######################
EPSILON = 0.1
actions = [0,1,2,3]
ACTIONS = dict()
ACTIONS[0] = 'U'
ACTIONS[1] = 'D'
ACTIONS[2] = 'R'
ACTIONS[3] = 'L'

GRID_SIZE   = (4,12)
START       = (3,0)
GOAL        = (3,11)
GAMMA       = 1
#########################################################
stateActionValueSarsa = np.zeros((4,12,4))
stateActionValueQ = np.zeros((4,12,4))
gridWorldReturn = -1*np.ones(GRID_SIZE)
cliffRow        = GRID_SIZE[0]-1
cliffColumn     = GRID_SIZE[1]-1
gridWorldReturn[cliffRow, 1:cliffColumn] = -100

states = []
for i in range(GRID_SIZE[0]):
    for j in range(GRID_SIZE[1]):
        states.append([i, j])

def next_state(state, action):
    i, j = state
    a_str = ACTIONS[action]
    if i==0 and a_str=='U':
        return (i,j)
    if i==GRID_SIZE[0]-1 and a_str=='D':
        return (i,j)
    if j == 0 and a_str == 'L':
        return (i, j)
    if j == GRID_SIZE[1]-1 and a_str == 'R':
        return (i, j)

    if a_str == 'U':
        return (i - 1, j)
    if a_str == 'D':
        return (i + 1, j)
    if a_str == 'R':
        return (i, j + 1)
    if a_str == 'L':
        return (i, j - 1)


def chooseAction(stateActionValue, state):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(actions)
    else:
        qs = []
        for action in actions:
            qs.append(stateActionValue[state[0], state[1], action])
        index = np.argmax(qs)

        return actions[index]


def Sarsa(alpha=0.5):
    rewards = 0.0
    currState = START
    currAction = chooseAction(stateActionValueSarsa, currState)
    while(currState!=GOAL):
        nextState   = next_state(currState, currAction)
        nextAction  = chooseAction(stateActionValueSarsa, nextState)
        reward      = gridWorldReturn[nextState[0], nextState[1]]
        rewards     += reward
        currQ       = stateActionValueSarsa[currState[0], currState[1], currAction]
        nextQ       = GAMMA*stateActionValueSarsa[nextState[0], nextState[1], nextAction]
        stateActionValueSarsa[currState[0], currState[1], currAction] += alpha*(reward + nextQ - currQ)
        currState   = nextState
        currAction  = nextAction

    return rewards

def chooseMaxQ(stateActionValue, state):
        qs = []
        for action in actions:
            qs.append(stateActionValue[state[0], state[1], action])
        index = np.argmax(qs)

        return qs[index]


def QLearning(alpha=0.5):
    rewards = 0.0
    currState = START
    while(currState!=GOAL):
        currAction  = chooseAction(stateActionValueQ, currState)
        reward      = gridWorldReturn[currState[0], currState[1]]
        rewards     += reward
        currQ       = stateActionValueQ[currState[0], currState[1], currAction]
        nextState   = next_state(currState, currAction)
        qmax = chooseMaxQ(stateActionValueQ, nextState)
        stateActionValueQ[currState[0], currState[1], currAction] += alpha*(reward + GAMMA*qmax - currQ)
        currState   = nextState

    return rewards


def main():
    nEpisode = 500
    runs = 50

    rewardsSarsa = np.zeros(nEpisode)
    rewardsQLearning = np.zeros(nEpisode)
    for run in range(0, runs):
        stateActionValueSarsa = np.zeros((4,12,4))
        stateActionValueQ = np.zeros((4,12,4))
        for i in range(nEpisode):
            rewardsSarsa[i] += Sarsa()
            rewardsQLearning[i] += QLearning()

    rewardsSarsa /= runs
    rewardsQLearning /= runs

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(range(nEpisode), rewardsSarsa, label='Sarsa')
    ax.plot(range(nEpisode), rewardsQLearning, label='Q Learning')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Sum of Rewards During Episode')
    plt.grid(which='major', color='black', linestyle='--')
    plt.legend()
    plt.ylim(-100, -10)
    plt.show()

if __name__ == '__main__':
    main()
