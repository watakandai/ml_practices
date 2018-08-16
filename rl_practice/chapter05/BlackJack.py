from __future__ import print_function

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get a new card
def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

def play(policyPlayerFn, initialState=None, initialAction=None):
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


def BlackJack(args):
    pass

def main(args):

    states, values, policies = BlacJack(args)

    x = np.random.random((100, 100))
    plt.imshow(x, cmap="PuRd")
    plt.colorbar()

    x = np.arange(-3, 3, 0.25)
    y = np.arange(-3, 3, 0.25)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X)+ np.cos(Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X, Y, Z) #<---ここでplot

    if args.saveimage == True:
        plt.savefig('figure.png')
    if args.showimage == True:
        plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Black Jack Function Approximation using Monte Carlo Estimation')
    parser.add_argument('--saveimage', '-s', action='store_true',
                        help='Save Image')
    parser.add_argument('--showimage', '-i', action='store_true',
                        help='Show Image')
    args = parser.parse_args()

    main(args)
