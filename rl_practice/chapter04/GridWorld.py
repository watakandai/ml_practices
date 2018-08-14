from __future__ import print_function

import argparse
import numpy as np

NUM_ACTIONS = 4

def calc_next_state(state, p, gridsize):
    size    = gridsize*gridsize
    b       = np.array(range(gridsize))
    if np.any(b[:]==state) and p==1:
        return state
    bound   = gridsize*b
    if np.any(bound[:]==state) and p==4:
        return state
    bound   = gridsize*(b+1)-1
    if np.any(bound[:]==state) and p==2:
        return state
    bound   = np.array(range(size-gridsize, size))
    if np.any(bound[:]==state) and p==3:
        return state

    if p==1:
        return state-gridsize
    if p==2:
        return state+1
    if p==3:
        return state+gridsize
    if p==4:
        return state-1


def calc_new_value(state, values, policy, gridsize):
    sum = 0
    for p in policy:
            next_state = calc_next_state(state, p, gridsize)
            sum += 1/NUM_ACTIONS *(-1 + values[next_state])
    return sum

def policy_evaluation(states, values, policy, gridsize, threshold=0.0000001):
    iteration = 1
    new_values = values.copy()
    while True:
        delta = 0
        print(iteration)
        for state in states:
            if not (state==0 or state==gridsize*gridsize-1):
                v       = values[state]
                new_v   = calc_new_value(state, values, policy, gridsize)
                new_values[state] = new_v

                delta = max(delta, v-new_v)
        if delta<threshold:
            break

        values = new_values.copy()
        print(np.reshape(values, (gridsize, gridsize)))
        iteration +=1

    return iteration

def policty_improvement():
    pass

def main():
    parse = argparse.ArgumentParser(description='Grid World Policty Iteration')
    parse.add_argument('--gridsize','-s', type=int, default=4,
                        help='Size of the Grid')
    args = parse.parse_args()

    size    = args.gridsize*args.gridsize
    States  = np.array(range(size))
    Values   = np.zeros(size)
    Policy  = np.array([1,2,3,4]) # up, right, bottom, left
    print('States')
    print(States)
    print('Initial Value')
    print(Values)
    print('Policy')
    print(Policy)

    iteration = policy_evaluation(States, Values, Policy, args.gridsize)
    print('Iteration: %i'%(iteration))



if __name__ == "__main__":
    main()
