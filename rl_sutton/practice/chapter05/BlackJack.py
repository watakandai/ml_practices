########################################################
#
# Black Jack Rule:
#   - 21超えないMaxを狙う．
#   - 絵札は全部10点とする
#   - エース(A)は 1点でも11点にもなりうる
#   - 今回はユーザーvsディーラーの1on1
#   - ディーラー，ユーザー共に2枚から始める
#   - その内，ディーラーの1枚は見えるものとする
#   - 始めから21があれば，"Natural"と呼び，ディーラーも21がなければ勝ち．あれば引き分け．．
#   - if wanted, request another card
#   - dealer's turn (hit under 16, stay over 17)
#
#   - cards are dealt from infinite deck
#   - Ace: 11とカウントしてBustにならない＝Usable = 11
#   ディーラー，プレーヤーの合計，エースの獲得はランダム
#
########################################################
from __future__ import print_function

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

################### GLOBAL CONSTANTS ###################
# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
actions = [ACTION_HIT, ACTION_STAND]

# policy for player
policyPlayer = np.zeros(22)
for i in range(12, 20):
    policyPlayer[i] = ACTION_HIT
policyPlayer[20] = ACTION_STAND
policyPlayer[21] = ACTION_STAND

# policy for dealer
policyDealer = np.zeros(22)
for i in range(12, 17):
    policyDealer[i] = ACTION_HIT
for i in range(17, 22):
    policyDealer[i] = ACTION_STAND
########################################################

# get a new card
def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# play a game
# @policyPlayerFn: specify policy for player
# @initialState: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initialAction: the initial action
def playBlackJack(policyPlayerFn, initialState=None, initialAction=None):
    # player status

    # sum of player
    playerSum = 0

    # trajectory of player
    playerTrajectory = []

    # whether player uses Ace as 11
    usableAcePlayer = False

    # dealer status
    dealerCard1 = 0
    dealerCard2 = 0
    usableAceDealer = False

    if initialState is None:
        # generate a random initial state

        numOfAce = 0

        # initialize cards of player
        while playerSum < 12:
            # if sum of player is less than 12, always hit
            card = getCard()

            # if get an Ace, use it as 11
            if card == 1:
                numOfAce += 1
                card = 11
                usableAcePlayer = True
            playerSum += card

        # if player's sum is larger than 21, he must hold at least one Ace, two Aces are possible
        if playerSum > 21:
            # use the Ace as 1 rather than 11
            playerSum -= 10

            # if the player only has one Ace, then he doesn't have usable Ace any more
            if numOfAce == 1:
                usableAcePlayer = False

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealerCard1 = getCard()
        dealerCard2 = getCard()

    else:
        # use specified initial state
        usableAcePlayer = initialState[0]
        playerSum = initialState[1]
        dealerCard1 = initialState[2]
        dealerCard2 = getCard()

    # initial state of the game
    state = [usableAcePlayer, playerSum, dealerCard1]

    # initialize dealer's sum
    dealerSum = 0
    if dealerCard1 == 1 and dealerCard2 != 1:
        dealerSum += 11 + dealerCard2
        usableAceDealer = True
    elif dealerCard1 != 1 and dealerCard2 == 1:
        dealerSum += dealerCard1 + 11
        usableAceDealer = True
    elif dealerCard1 == 1 and dealerCard2 == 1:
        dealerSum += 1 + 11
        usableAceDealer = True
    else:
        dealerSum += dealerCard1 + dealerCard2

    # game starts!

    # player's turn
    while True:
        if initialAction is not None:
            action = initialAction
            initialAction = None
        else:
            # get action based on current sum
            action = policyPlayerFn(usableAcePlayer, playerSum, dealerCard1)

        # track player's trajectory for importance sampling
        playerTrajectory.append([(usableAcePlayer, playerSum, dealerCard1), action])

        if action == ACTION_STAND:
            break
        # if hit, get new card
        playerSum += getCard()

        # player busts
        if playerSum > 21:
            # if player has a usable Ace, use it as 1 to avoid busting and continue
            if usableAcePlayer == True:
                playerSum -= 10
                usableAcePlayer = False
            else:
                # otherwise player loses
                return state, -1, playerTrajectory

    # dealer's turn
    while True:
        # get action based on current sum
        action = policyDealer[dealerSum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = getCard()
        if new_card == 1 and dealerSum + 11 < 21:
            dealerSum += 11
            usableAceDealer = True
        else:
            dealerSum += new_card
        # dealer busts
        if dealerSum > 21:
            if usableAceDealer == True:
            # if dealer has a usable Ace, use it as 1 to avoid busting and continue
                dealerSum -= 10
                usableAceDealer = False
            else:
            # otherwise dealer loses
                return state, 1, playerTrajectory

    # compare the sum between player and dealer
    if playerSum > dealerSum:
        return state, 1, playerTrajectory
    elif playerSum == dealerSum:
        return state, 0, playerTrajectory
    else:
        return state, -1, playerTrajectory

def policyFn(usableAcePlayer, playerSum, dealerCard):
    usableAcePlayer = int(usableAcePlayer)
    playerSum -= 12
    dealerCard -= 1
    actionValues = []
    for action in actions:
        statevalue = stateValue[playerSum, dealerCard, usableAcePlayer, action]
        paircount = stateActionPairCount[playerSum, dealerCard, usableAcePlayer, action]
        actionValues.append(statevalue/paircount)
    index = np.argmax(actionValues)
    return actions[index]


stateValue = np.zeros((10, 10, 2, 2))
stateActionPairCount = np.ones((10, 10, 2, 2))
def Example5_3(args):
    global stateValue, stateActionPairCount
    num_episode = args.num_episode
    for episode in range(num_episode):
        if episode % 1000 == 0:
            print('episode:', episode)
        initialState = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initialAction = np.random.choice(actions)
        _, reward, trajectory = playBlackJack(policyFn, initialState, initialAction)
        for (usableAce, playerSum, dealerCard), action in trajectory:
            usableAce = int(usableAce)
            playerSum -= 12
            dealerCard -= 1
            stateValue[playerSum, dealerCard, usableAce, action] += reward
            stateActionPairCount[playerSum, dealerCard, usableAce, action] += 1

    return stateValue / stateActionPairCount


# print the state value
figureIndex = 0
def prettyPrint(data, tile, zlabel='reward'):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(tile)
    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []
    for i in range(12, 22):
        for j in range(1, 11):
            axisX.append(i)
            axisY.append(j)
            axisZ.append(data[i - 12, j - 1])
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('player sum')
    ax.set_ylabel('dealer showing')
    ax.set_zlabel(zlabel)



def main(args):
    stateActionValues = Example5_3(args)

    stateValueUsableAce = np.zeros((10, 10))
    stateValueNoUsableAce = np.zeros((10, 10))
    # get the optimal policy
    actionUsableAce = np.zeros((10, 10), dtype='int')
    actionNoUsableAce = np.zeros((10, 10), dtype='int')
    for i in range(10):
        for j in range(10):
            stateValueNoUsableAce[i, j] = np.max(stateActionValues[i, j, 0, :])
            stateValueUsableAce[i, j] = np.max(stateActionValues[i, j, 1, :])
            actionNoUsableAce[i, j] = np.argmax(stateActionValues[i, j, 0, :])
            actionUsableAce[i, j] = np.argmax(stateActionValues[i, j, 1, :])
    prettyPrint(stateValueUsableAce, 'Optimal state value with usable Ace')
    prettyPrint(stateValueNoUsableAce, 'Optimal state value with no usable Ace')
    prettyPrint(actionUsableAce, 'Optimal policy with usable Ace', 'Action (0 Hit, 1 Stick)')
    prettyPrint(actionNoUsableAce, 'Optimal policy with no usable Ace', 'Action (0 Hit, 1 Stick)')
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Black Jack Function Approximation using Monte Carlo Estimation')
    parser.add_argument('--num_episode', '-n', type=int, default=500000,
                        help='Number of Episodes')
    parser.add_argument('--saveimage', '-s', action='store_true',
                        help='Save Image')
    parser.add_argument('--showimage', '-i', action='store_true',
                        help='Show Image')
    args = parser.parse_args()

    main(args)
