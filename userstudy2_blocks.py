'''
Code for Section 6 User Study
This code runs the simulation for the study in 6.2
'''

import numpy as np
import random
import copy
from matplotlib import pyplot as plt
import argparse
import pickle

# by default runs the simulation opaque algorithm with learning rate 0.5
# get parameters for simulation
parser = argparse.ArgumentParser()
parser.add_argument('--alg', default="ours", help='which algorithm to run. options are ours and trans')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate for the simulation') 
args = parser.parse_args()


# formalize the stochastic bayesian game
class TowerSBG:

     # initialization
    def __init__(self, T, lr):

        # time horizon
        self.T = T
        # learning rate (does not apply to bayes)
        self.lr = lr
        # augmented state space
        # (timestep t, state s, belief b)
        self.states = []
        for belief in np.linspace(0, 1.0, 11):
            self.states.append((0, (-1, -1, -1, -1, -1, -1), round(belief, 1)))
        for block1 in range(6):
            for block2 in range(6):
                for belief in np.linspace(0, 1.0, 11):
                    tower = (block1, block2, -1, -1, -1, -1)
                    self.states.append((1, tower, round(belief, 1)))
        for block1 in range(6):
            for block2 in range(6):
                for block3 in range(6):
                    for block4 in range(6):
                        for belief in np.linspace(0, 1.0, 11):
                            tower = (block1, block2, block3, block4, -1, -1)
                            self.states.append((2, tower, round(belief, 1)))
        for block1 in range(6):
            for block2 in range(6):
                for block3 in range(6):
                    for block4 in range(6):
                        for block5 in range(6):
                             for block6 in range(6):
                                for belief in np.linspace(0, 1.0, 11):
                                    tower = (block1, block2, block3, block4, block5, block6)
                                    self.states.append((3, tower, round(belief, 1)))
        # some good use of for loops with keep this scaling up
        # currently the tower can only hold a max of six blocks

        # action space
        # 4 blocks to choose from for capable, 2 for confused
        # action space for the confused robot
        self.actions_r1 = range(2) 
        # action space for the capable robot
        self.actions_r2 = range(4) 
        # action space for the human
        self.actions_h = range(4)

    # dynamics
    def f(self, s, ah, ar):
        timestep = s[0]
        if timestep == 0:
            state = (ah, ar, -1, -1, -1, -1)
        if timestep == 1:
            state = (s[1][0], s[1][1], ah, ar, -1, -1)
        if timestep == 2:
            state = (s[1][0], s[1][1], s[1][2], s[1][3], ah, ar)
        belief = s[2]
        if belief > 0.01 and belief < 0.99:
            if ar >= 3:
                belief = min([0.9, belief + self.lr])
            else:
                belief = max([0.1, belief - self.lr])
        return (timestep+1, state, round(belief,1))

    # reward function
    def reward(self, s):
        timestep, state = s[0], s[1]
        matching_bonus = 5.0 # bonus for choosing the same block
        height_bonus = 1.0  # bonus for choosing a big block
        rewards = [0, 0, 0]
        for idx, pair in enumerate(((0, 1), (2, 3), (4, 5))):
            if state[pair[0]] == state[pair[1]]:
                rewards[idx] = +matching_bonus
            else:
                rewards[idx] = -matching_bonus
            if state[pair[0]] > 1:
                rewards[idx] += height_bonus
            if state[pair[1]] > 1:
                rewards[idx] += height_bonus
        if timestep == 0:
            return 0.
        if timestep == 1:
            return rewards[0]
        if timestep == 2:
            return rewards[0] + rewards[1]
        if timestep == 3:
            return rewards[0] + rewards[1] + rewards[2]

    # bonus reward for transparency
    def bonus_reward(self, a):
        ah, ar1, ar2 = a
        difference = 3*abs(ar1 - ar2) 
        return difference
    
    # modified Harsanyi-Bellman Ad Hoc Coordination
    # see equations (4)-(6) in paper
    # pi maps state to optimal human and robot actions
    def value_iteration(self, args):
        V1 = {s: 0 for s in self.states}
        pi = {s: None for s in self.states}
        for _ in range(self.T+1):
            V = V1.copy()
            for s in self.states:
                if s[0] >= self.T:
                    V1[s] = self.reward(s)
                    continue
                v_next_max = -np.inf
                for ah in self.actions_h:
                    for ar1 in self.actions_r1:
                        for ar2 in self.actions_r2:
                            s1 = self.f(s, ah, ar1)
                            s2 = self.f(s, ah, ar2)
                            eV1 = (1-s[2]) * V[s1]
                            eV2 = s[2] * V[s2]
                            if args.alg == "trans":
                                eV1 += 1.0 * self.bonus_reward([ah, ar1, ar2])
                            if eV1 + eV2 > v_next_max:
                                v_next_max = eV1 + eV2
                                pi[s] = [ah, ar1, ar2]
                V1[s] = self.reward(s) + v_next_max 
        return pi, V1
    
def main(args):

    # get the simulation parameters
    T = 3
    lr = args.lr 

    # get optimal policy for human and robot
    tower_sbg = TowerSBG(T, lr)
    pi, V = tower_sbg.value_iteration(args)

    ## save pi for use on actual robot arm
    ## save result
    # pickle.dump(pi, open("res/pi-b4-t-" + str(T) + "-lr-" + str(args.lr) + args.alg +  ".pkl", 'wb'))
    # print("[*] saved: ", "res/pi-b4-t-" + str(T) + "-lr-" + str(args.lr) + args.alg +  ".pkl")
    

    # everything below is just for testing

    # timestep 0, empty tower, initial belief
    init_state = (0, (-1, -1, -1, -1, -1, -1), 0.5)
    
    # rollout policy with robot type 1 (confused robot)
    s1 = copy.deepcopy(init_state)
    print(tower_sbg.reward(s1))
    print("[*] type 1 - Confused Robot")
    for t in range(tower_sbg.T):
        astar = pi[s1]
        print(s1, astar)
        print(tower_sbg.reward(s1))
        # Rational Human
        s1 = tower_sbg.f(s1, astar[0], astar[1])
    print(s1)
    print(tower_sbg.reward(s1))

    # rollout policy with robot type 2 (capable robot)
    s2 = copy.deepcopy(init_state)
    print(tower_sbg.reward(s2))
    print("[*] type 2 - Capable Robot")
    for t in range(tower_sbg.T):
        astar = pi[s2]
        print(s2, astar)
        print(tower_sbg.reward(s2))
        # Rational Human
        s2 = tower_sbg.f(s2, astar[0], astar[2])
    print(s2)
    print(tower_sbg.reward(s2))



main(args)