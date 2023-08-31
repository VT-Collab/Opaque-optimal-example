'''
Code for Section 5 What Conditions Lead to Opaque Robots?
This code runs the simulation for the 1D example in the paper.
Results are saved in the sim1 folder.
Plotter.py in sim1 folder can be used to plot the results.
'''

import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import argparse
import pickle


# by default runs the simulation for 10 timesteps
# get parameters for simulation
parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=10, help='time horizon')
args = parser.parse_args()


# formalize the stochastic bayesian game
class ExampleSBG:

     # initialization
    def __init__(self, T):

        # time horizon
        self.T = T
        # augmented state space
        # (timestep t, state s, belief b)
        self.states = []
        for t in range(self.T):
            for s in np.linspace(0, 2.0, 21):
                for b in np.linspace(0, 1.0, 11):
                    augmented_state = (t, round(s,1), round(b,1))
                    self.states.append(augmented_state)
        # action space
        # action space for the confused robot
        self.actions_r1 = [-0.1]
        # action space for the capable robot
        self.actions_r2 = [-0.1, 0.1]
        # action space for the human
        self.actions_h = [-0.1, 0.0, 0.1]
        # initialize policy, needed for boltzmann model
        self.pi = {s: None for s in self.states}

    # dynamics
    def f(self, s, ah, ar):
        timestep = s[0]
        # both human and robot action move the system
        state = s[1] + ah + ar
        state = min([2.0, state])
        state = max([0.0, state])
        belief = s[2]
        if belief > 0.01 and belief < 0.99:
            # if both robots take same action,
            # cannot learn anything, belief stays same
            if abs(self.pi[s][1] - self.pi[s][2]) < 0.01:
                belief = s[2]
            # if robot action matches the action of type1 robot (confused)
            # we must be working with type1
            elif abs(ar - self.pi[s][1]) < 0.01:
                belief = 0.0
            # if robot action matches the action of type2 robot (capable)
            # we must be working with type2
            elif abs(ar - self.pi[s][2]) < 0.01:
                belief = 1.0
            else:
                print("we should not be here")
        return (timestep+1, round(state,1), round(belief,1))

    # reward function
    def reward(self, s):
        timestep, state = s[0], s[1]
        if timestep == self.T-1:
            if state == 0.0:
                return +1.0
            if state == 2.0:
                return +2.0
        return 0.0

    # modified Harsanyi-Bellman Ad Hoc Coordination
    # see equations (4)-(6) in paper
    # pi maps state to optimal human and robot actions
    def value_iteration(self):
        V1 = {s: 0 for s in self.states}
        pi1 = {s: None for s in self.states}
        for _ in range(self.T+1):
            V = V1.copy()
            for s in self.states:
                if s[0] == self.T-1:
                    V1[s] = self.reward(s)
                    continue
                v_next_max = -np.inf
                for ah in self.actions_h:
                    for ar1 in self.actions_r1:
                        for ar2 in self.actions_r2:
                            self.pi[s] = [ah, ar1, ar2]
                            s1 = self.f(s, ah, ar1)
                            s2 = self.f(s, ah, ar2)
                            eV1 = (1-s[2]) * V[s1]
                            eV2 = s[2] * V[s2]
                            if eV1 + eV2 > v_next_max:
                                v_next_max = eV1 + eV2
                                pi1[s] = [ah, ar1, ar2]
                V1[s] = self.reward(s) + v_next_max
        self.pi = pi1
        return pi1, V1


# generate policies for the random human
def rand_human_policy(example_sbg, action='none'):
    pi_h = {}
    if action == 'none':
        for state in example_sbg.states:
            pi_h[state] = random.choice(example_sbg.actions_h)
    else:
        for state in example_sbg.states:
            pi_h[state] = action
    return pi_h


# check if an initial state is opaque
def check_opaque(init_state, example_sbg, pi, human_type="rational", N=1000):

    # if rational only need one iteration
    # if random we need N iterations to try random policies
    if human_type == "rational":
        N = 1

    # main loop
    for iteration in range(N):
        # get a random human policy
        if iteration == 0:
            # this seems adversarial
            pi_h = rand_human_policy(example_sbg, min(example_sbg.actions_h))
        elif iteration == 1:
            # this seems adversarial
            pi_h = rand_human_policy(example_sbg, max(example_sbg.actions_h))
        elif iteration == 2:
            # this seems adversarial
            pi_h = rand_human_policy(example_sbg, 0.0)
        else:
            # ok let's try totally random
            pi_h = rand_human_policy(example_sbg)

        # rollout policy with robot type 1
        s1 = copy.deepcopy(init_state)
        for t in range(example_sbg.T-1):
            astar = pi[s1]
            # Rational Human
            if human_type == "rational":
                s1 = example_sbg.f(s1, astar[0], astar[1])
            # Random Human
            if human_type == "random":
                ah = pi_h[s1]
                s1 = example_sbg.f(s1, ah, astar[1])

        # rollout policy with robot type 2
        s2 = copy.deepcopy(init_state)
        for t in range(example_sbg.T-1):
            astar = pi[s2]
            # Rational Human
            if human_type == "rational":
                s2 = example_sbg.f(s2, astar[0], astar[2])
            # Random Human
            if human_type == "random":
                ah = pi_h[s2]
                s2 = example_sbg.f(s2, ah, astar[2])

        # if beliefs are different then not opaque
        if abs(s1[2] - s2[2]) > 1e-3:
            return False

    # if we made it here then it is opaque
    return True


def main(args):

    # get the simulation parameters
    T = args.t

    # keep track of which states are opaque
    opaque_states = {}

    # get optimal policy for human and robot
    block1d = ExampleSBG(T)
    pi, V = block1d.value_iteration()

    # check all my states to see if opaque
    for b0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for s0 in np.linspace(0, 2.0, 21):

            # choose initial augmented state
            # (timestep t, state s, belief b)
            augmented_state = (0, round(s0,1), round(b0,1))

            # check if state is rationally / fully opaque
            rationally_opaque = check_opaque(augmented_state, block1d, pi, human_type="rational", N=1)
            if rationally_opaque == False:
                fully_opaque = False
            else:
                fully_opaque = check_opaque(augmented_state, block1d, pi, human_type="random", N=100)
            opaque_states[str(augmented_state)] = (augmented_state, rationally_opaque, fully_opaque)

    # save result
    pickle.dump(opaque_states, open("sim1/bayes-t-" + str(T) + ".pkl", 'wb'))
    print("[*] saved: ", "sim1/bayes-t-" + str(T) + ".pkl")



main(args)
