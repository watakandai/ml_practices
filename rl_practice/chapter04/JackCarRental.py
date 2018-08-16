from __future__ import print_function

import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rentnum_probaility(n, lam):
    return (lam**n/math.factorial(n))*math.exp(-lam)


def calcValue(state, credit, action, action_cost, values, max_car, lambdas, gamma):
    returns = 0.0

    returns -= action_cost * abs(action)
    for request1 in range(max_car):
        for request2 in range(max_car):
            next_state1 = int(min(state[0] - action, max_car))
            next_state2 = int(min(state[1] + action, max_car))

            rentable_num1 = min(request1, next_state1)
            rentable_num2 = min(request2, next_state2)

            reward = credit*(rentable_num1+rentable_num2)
            next_state1 -= rentable_num1
            next_state2 -= rentable_num2

            prob =  rentnum_probaility(next_state1, lambdas["Request1"]) * \
                    rentnum_probaility(next_state2, lambdas["Request2"])

            returnable_num1 = int(max_car-next_state1)
            returnable_num2 = int(max_car-next_state2)

            '''
            print('next state 1: {}'.format(next_state1))
            print('next state 2: {}'.format(next_state2))
            print('rentable 1: {}'.format(rentable_num1))
            print('rentable 2: {}'.format(rentable_num2))
            print('reward: {}'.format(reward))
            print('prob: {}'.format(prob))
            print('returnable_num1: {}'.format(returnable_num1))
            print('returnable_num2: {}'.format(returnable_num2))
            '''
            for return1 in range(returnable_num1):
                for return2 in range(returnable_num2):
                    state1     = next_state1
                    state2     = next_state2
                    total_prob =    rentnum_probaility(return1, lambdas["Return1"]) * \
                                    rentnum_probaility(return2, lambdas["Return2"]) * prob
                    state1     += return1
                    state2     += return2
                    # print('return1: {}'.format(return1))
                    # print('return2: {}'.format(return2))
                    # print('state1: {}'.format(state1))
                    # print('state2: {}'.format(state2))
                    returns += total_prob * (reward + gamma*values[state1, state2] )

    return returns

def JackCarRental(args):
    # Set Variables
    max_car     = args.max_car # No. of Cars: 0~max_car
    credit      = args.credit
    action_cost = args.action_cost
    max_action  = args.max_action
    gamma       = args.gamma
    lambdas     = {"Request1":3, "Request2":4, "Return1":3, "Return2":2}
    threshold   = 0.001
    # Print Variables
    print('----------------------Variables-------------------------')
    print('Max Cars: {}'.format(max_car))
    print('Credit for 1 Car Rental: {}'.format(credit))
    print('Cost for 1 Action: {}'.format(action_cost))
    print('Max No. of Cars that Can be Moved at 1 night: {}'.format(max_action))

    print('Gamma: {}'.format(gamma))
    print('Lambda: {0}, {1}, {2}, {3}'.format(lambdas["Request1"], lambdas["Request2"], lambdas["Return1"], lambdas["Return2"]))
    print('--------------------------------------------------------')

    # init states, values & policies
    states = []
    for i in range(max_car+1):
        for j in range(max_car+1):
            states.append([i, j])
    values      = np.zeros((max_car+1, max_car+1))
    new_values  = values.copy()
    policies    = np.zeros((max_car+1, max_car+1))
    new_policies = np.zeros((max_car+1, max_car+1))
    actions     = np.array(range(-max_action, max_action+1))
    '''
    print(states)
    print(values)
    print(policies)
    print(actions)
    '''

    record_values   = []
    record_actions  = []

    print('-----------------Start Simulation------------------')
    while True:
        # Policy Evaluation
        while True:
            for (i, j)  in states:
                action  = policies[i,j]
                new_values[i,j] = calcValue([i,j], credit, action, action_cost, values, max_car, lambdas, gamma)

            if np.sum(np.abs(new_values-values)) < threshold:
                values  = new_values.copy()
                record_values.append(values)
                print("=====State Value=====")
                print(values.astype('int64'))
                break

            print(values.astype('int64'))
            print('New Value')
            print(new_values.astype('int64'))
            values = new_values.copy()

        # Policy Improvement
        policy_stable = True
        for (i, j)  in states:
            actionReturns = []
            for action in actions:
                new_action = calcValue([i,j], credit, action, action_cost, values, max_car, lambdas, gamma)
                actionReturns.append(new_action)
            bestAction = np.argmax(actionReturns)
            new_policies[i,j] = actions[bestAction]

        if new_policies != policies:
            policy_stable = False
        policies = new_policies
        record_actions.append(policies)

        if policy_stable:
            break


    return states, record_values, record_actions

def main(args):
    states, values, policies = JackCarRental(args)
    with open('values.pkl', 'wb') as fvalue:
        pickle.dump(values,fvalue)
    with open('policies', 'wb') as fpolicy:
        pickle.dump(policies, fpolicy)
    best_value  = values[-1]
    best_policy = policies[-1]

    print('BEST VALUE: ------')
    print(best_value)
    print('BEST POLICY: ------')
    print(best_policy)

    max_car     = args.max_car # No. of Cars: 0~max_car

    x = np.array(range(max_car+1))
    y = np.array(range(max_car+1))
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X, Y, best_value) #<---ここでplot

    if args.saveimage == True:
        plt.savefig('figure.png')
    if args.showimage == True:
        plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Jacks Car Rental')
    parser.add_argument('--max_car', '-mc', type=int, default=20,
                        help='Max Number of Cars at Each Shop')
    parser.add_argument('--gamma', '-g', type=int, default=0.9,
                        help='Discount Rate')
    # Array as an Argument
    # https://stackoverflow.com/questions/15753701/argparse-option-for-passing-a-list-as-option
    parser.add_argument('--credit', '-c', type=int, default=10,
                        help='Credit for 1 Car Rental')
    parser.add_argument('--action_cost', '-a', type=int, default=2,
                        help='Cost of each Action')
    parser.add_argument('--max_action', '-ma', type=int, default=5,
                        help='Max Number of Movable Cars')
    parser.add_argument('--saveimage', '-s', action='store_true',
                        help='Save Image')
    parser.add_argument('--showimage', '-i', action='store_true',
                        help='Show Image')
    args = parser.parse_args()

    main(args)
