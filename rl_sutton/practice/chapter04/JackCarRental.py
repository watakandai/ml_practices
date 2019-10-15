from __future__ import print_function

from math import *
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


poissonBackup = dict()
def poisson(n, lam):
    global poissonBackup
    key = n * 10 + lam
    if key not in poissonBackup.keys():
        poissonBackup[key] = exp(-lam) * pow(lam, n) / factorial(n)
    return poissonBackup[key]


def calcValue(state, credit, action, action_cost, values, max_car, lambdas, gamma):
    returns = 0.0

    returns -= action_cost * abs(action)
    # for request1 in range(max_car):
     #    for request2 in range(max_car):
    for request1 in range(0,11):
        for request2 in range(0,11):
            next_state1 = int(min(state[0] - action, max_car))
            next_state2 = int(min(state[1] + action, max_car))

            rentable_num1 = min(request1, next_state1)
            rentable_num2 = min(request2, next_state2)

            reward = credit*(rentable_num1+rentable_num2)
            next_state1 -= rentable_num1
            next_state2 -= rentable_num2

            prob =  poisson(request1, lambdas["Request1"]) * \
                    poisson(request2, lambdas["Request2"])

            returnable_num1 = int(max_car-next_state1)
            returnable_num2 = int(max_car-next_state2)

            constantReturnedCars = True
            if constantReturnedCars:
                state1 = min(next_state1+lambdas["Return1"], max_car)
                state2 = min(next_state2+lambdas["Return2"], max_car)
                returns += prob * (reward + gamma * values[state1, state2])
            else:
                for return1 in range(returnable_num1):
                    for return2 in range(returnable_num2):
                        state1     = next_state1
                        state2     = next_state2
                        total_prob =    poisson(return1, lambdas["Return1"]) * \
                                        poisson(return2, lambdas["Return2"]) * prob
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
    print('----------------Variables & Constants -----------------')
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

            # print('Value:')
            # print(values.astype('int64'))
            # print('New Value')
            # print(new_values.astype('int64'))
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

        policyChanges = np.sum(new_policies != policies)
        print('Policy for {} states changed'.format(policyChanges))
        if policyChanges == 0:
            policies = new_policies
            break
        policies = new_policies
        record_actions.append(policies)

    return states, record_values, record_actions

# axes for printing use
AxisXPrint = []
AxisYPrint = []
for i in range(0, 20 + 1):
    for j in range(0, 20 + 1):
        AxisXPrint.append(i)
        AxisYPrint.append(j)

figureIndex = 0
def prettyPrint(states, data, labels):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    ax = fig.add_subplot(111, projection='3d')
    AxisZ = []
    for i, j in states:
        AxisZ.append(data[i, j])
    ax.scatter(AxisXPrint, AxisYPrint, AxisZ)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

def main(args):
    if args.justprint:
        states = pickle.load(open('states.pkl', 'rb'))
        policies = pickle.load(open('policies.pkl', 'rb'))
        values = pickle.load(open('values.pkl', 'rb'))
        prettyPrint(states, policies[-1], ['# of cars in first location', '# of cars in second location', '# of cars to move during night'])
        prettyPrint(states, values[-1], ['# of cars in first location', '# of cars in second location', 'expected returns'])
        plt.show()
    else:
        states, values, policies = JackCarRental(args)
        with open('states.pkl', 'wb') as fstate:
            pickle.dump(states, fstate)
        with open('values.pkl', 'wb') as fvalue:
            pickle.dump(values,fvalue)
        with open('policies.pkl', 'wb') as fpolicy:
            pickle.dump(policies, fpolicy)
        best_value  = values[-1]
        best_policy = policies[-1]

        print('BEST VALUE: ------')
        print(best_value)
        print('BEST POLICY: ------')
        print(best_policy)

        prettyPrint(states, policies[-1], ['# of cars in first location', '# of cars in second location', '# of cars to move during night'])
        prettyPrint(states, values[-1], ['# of cars in first location', '# of cars in second location', 'expected returns'])

        plt.savefig('figure.png')
        plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Jacks Car Rental')
    parser.print_help()
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
    parser.add_argument('--justprint', '-j', action='store_true',
                        help='Just Show Image from pickle file')
    args = parser.parse_args()

    main(args)
