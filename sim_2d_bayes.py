'''
Code for Section 5 What Conditions Lead to Opaque Robots?
This code runs the simulation for the 2D example in the paper.
Results are saved in the sim2 folder.
Plotter.py in sim2 folder can be used to plot the results.
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
parser.add_argument('--t', type=int, default=10, help="time horizon")
args = parser.parse_args()

# formalize the stochastic bayesian game
class RobotArmSBG:

     # initialization
    def __init__(self, T):

        # time horizon
        self.T = T
        # augmented state space
        # (timestep t, state x, state y, belief b)
        self.states = []
        for t in range(self.T):
            for sx in np.linspace(0, 1.0, 11):
                for sy in np.linspace(0, 1.0, 11):
                    for b in np.linspace(0, 1.0, 11):
                        augmented_state = (t, round(sx,1), round(sy,1), round(b,1))
                        self.states.append(augmented_state)
        # action space
        # action space for the confused robot
        self.actions_r1 = ((-0.1, 0.), (0., -0.1))
        # action space for the capable robot
        self.actions_r2 = ((-0.1, 0.), (0., -0.1), (+0.1, 0.), (0., +0.1))
        # action space for the human
        self.actions_h =  ((-0.1, 0.), (0., -0.1), (+0.1, 0.), (0., +0.1))
        # initialize policy, needed for boltzmann model
        self.pi = {s: None for s in self.states}

    # dynamics
    def f(self, s, ah, ar):
        timestep = s[0]
        statex = s[1] + ah[0] + ar[0]
        statey = s[2] + ah[1] + ar[1]
        statex = min([1.0, statex])
        statex = max([0.0, statex])
        statey = min([1.0, statey])
        statey = max([0.0, statey])
        belief = s[3]
        if belief > 0.01 and belief < 0.99:
            # if both robots take same action,
            # cannot learn anything, belief stays same
            if abs(self.pi[s][1][0] - self.pi[s][2][0]) < 0.01 and abs(self.pi[s][1][1] - self.pi[s][2][1]) < 0.01:
                belief = s[3]
            # if robot action matches the action of type1
            # we must be working with type1
            elif ar[0] < -0.01 or ar[1] < -0.01:
                belief = 0.0
            # if robot action matches the action of type2
            # we must be working with type2
            elif ar[0] > +0.01 or ar[1] > +0.01:
                belief = 1.0
            else:
                print("we should not be here")
        return (timestep+1, round(statex,1), round(statey,1), round(belief,1))

    # reward function
    def reward(self, s):
        timestep, statex, statey = s[0], s[1], s[2]
        if timestep == self.T-1:
            if statex == +0.0 and statey == +0.0:
                return +1.0
            if statex == +1.0 and statey == +0.0:
                return +2.0 
            if statex == +1.0 and statey == +1.0:
                return +3.0                        
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
                            eV1 = (1-s[3]) * V[s1]
                            eV2 = s[3] * V[s2]
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
        # its not clear to me which policies are adversarial...
        pi_h = rand_human_policy(example_sbg)

        # rollout policy with robot type 1 (confused robot)
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

        # rollout policy with robot type 2 (capable robot)
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
        if abs(s1[3] - s2[3]) > 1e-3:
            return False

    # if we made it here then it is opaque
    return True


def main(args):

    # get the simulation parameters
    T = args.t

    # keep track of which states are opaque
    opaque_states = {}

    # get optimal policy for human and robot
    block2d = RobotArmSBG(T)
    pi, V = block2d.value_iteration()

    # check all my states to see if opaque
    for b0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for sx in np.linspace(0, 1.0, 11):
            for sy in np.linspace(0, 1.0, 11):

                # choose initial augmented state
                # (timestep t, state s, belief b)
                augmented_state = (0, round(sx,1), round(sy,1), round(b0,1))

                # check if state is rationally / fully opaque
                rationally_opaque = check_opaque(augmented_state, block2d, pi, human_type="rational", N=1)
                if rationally_opaque == False:
                    fully_opaque = False
                else:
                    fully_opaque = check_opaque(augmented_state, block2d, pi, human_type="random", N=100)
                opaque_states[str(augmented_state)] = (augmented_state, rationally_opaque, fully_opaque)


    # save result
    pickle.dump(opaque_states, open("sim2/bayes-t-" + str(T) + ".pkl", 'wb'))
    print("[*] saved: ", "sim2/bayes-t-" + str(T) + ".pkl")


main(args)

