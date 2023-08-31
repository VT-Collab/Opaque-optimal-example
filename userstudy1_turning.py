'''
Code for Section 6 User Study
This code runs the simulation for the study in 6.1
The scenario here is Turning
'''

import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import argparse
import pickle

# by default runs the simulation opaque algorithm
# get parameters for simulation
parser = argparse.ArgumentParser()
parser.add_argument('--alg', default="ours", help='which algorithm to run. options are ours and trans')
args = parser.parse_args()


# formalize the stochastic bayesian game
class TurningSBG:

     # initialization
    def __init__(self):

        # time horizon
        self.T = 4
        # augmented state space
        # (timestep t, state x, state y, belief b)
        self.states = []
        for t in range(self.T):
            for angle in np.linspace(-3.0, 3.0, 13):
                for speed in np.linspace(0, 6.0, 7):
                    for b in np.linspace(0, 1.0, 3):
                        augmented_state = (t, round(angle,1), round(speed,1), round(b,1))
                        self.states.append(augmented_state)
        # action space
        # action space for the confused robot
        self.actions_r1 = ((0., 1.),)
        # action space for the capable robot
        self.actions_r2 = ((0., 1.), (-0.5, 0.), (+0.5, 0.))
        # action space for the human
        self.actions_h =  ((0., 1.), (-0.5, 0.), (+0.5, 0.))
        # initialize policy, needed for boltzmann model
        self.pi = {s: None for s in self.states}

    # dynamics
    def f(self, s, ah, ar):
        timestep = s[0]
        angle = s[1] + ah[0] + ar[0]
        speed = s[2] + ah[1] + ar[1]
        belief = s[3]
        angle = min([+3.0, angle])
        angle = max([-3.0, angle])
        speed = min([+6.0, speed])
        speed = max([-6.0, speed])
        if belief > 0.01 and belief < 0.99:
            # if both robots take same action,
            # cannot learn anything, belief stays same
            if abs(self.pi[s][1][0] - self.pi[s][2][0]) < 0.01 and abs(self.pi[s][1][1] - self.pi[s][2][1]) < 0.01:
                belief = s[3]
            # if robot action matches the action of type2
            # we must be working with type2
            elif abs(ar[0]) > 0.01:
                belief = 1.0
            else:
                belief = 0.0
        return (timestep+1, round(angle,1), round(speed,1), round(belief,1))

    # reward function
    def reward(self, s):
        timestep, angle, speed = s[0], s[1], s[2]
        if angle > 2.5:
            return +15
        else:
            return speed

    # bonus reward for transparency
    def bonus_reward(self, a):
        ah, ar1, ar2 = a
        difference = abs(ar1[0] - ar2[0]) + abs(ar1[1] - ar2[1])
        return difference

    # modified Harsanyi-Bellman Ad Hoc Coordination
    # see equations (4)-(6) in paper
    # pi maps state to optimal human and robot actions
    def value_iteration(self, args):
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
                            if args.alg == "trans":
                                eV1 += 1.0 * self.bonus_reward([ah, ar1, ar2])
                            if eV1 + eV2 > v_next_max:
                                v_next_max = eV1 + eV2
                                pi1[s] = [ah, ar1, ar2]
                V1[s] = self.reward(s) + v_next_max
        self.pi = pi1
        return pi1, V1


def main(args):

    # get optimal policy for human and robot
    env = TurningSBG()
    pi, V = env.value_iteration(args)

    # choose initial augmented state
    # (timestep t, state s, belief b)
    init_state = (0, 0., 0., 0.5)

    # rollout policy with robot type 1
    s1 = copy.deepcopy(init_state)
    print("[*] type 1")
    for t in range(env.T-1):
        astar = pi[s1]
        s1 = env.f(s1, astar[0], astar[1])
        print(s1, astar[1], astar[2])

    # rollout policy with robot type 1
    s2 = copy.deepcopy(init_state)
    print("[*] type 2")
    for t in range(env.T-1):
        astar = pi[s2]
        s2 = env.f(s2, astar[0], astar[2])
        print(s2, astar[1], astar[2])


main(args)
