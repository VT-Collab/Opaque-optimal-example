import numpy as np
import random
import copy
import argparse


# by default runs the example "Optimal Robots can be Fully Opaque"
# if --example rationally, then runs example "Optimal Robots can be Rationally Opaque"
parser = argparse.ArgumentParser()
parser.add_argument('--example', default="fully",
                    help='options are fully and rationally')
args = parser.parse_args()


# formalize the stochastic bayesian game
class ExampleSBG:

     # initialization
    def __init__(self):

        # time horizon
        self.T = 5
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
        self.actions_h = [-0.2, 0.0, 0.2]

    # dynamics
    def f(self, s, ah, ar):
        timestep = s[0]
        # both human and robot action move the system
        state = s[1] + ah + ar
        state = min([2.0, state])
        state = max([0.0, state])
        belief = s[2]
        # if robot moves right, human becomes more convinced robot is capable
        # otherwise human becomes more convinced robot is confused
        if ar > 0.0:
            belief = min([1.0, belief + 0.1])
        else:
            belief = max([0.0, belief - 0.1])
        return (timestep+1, round(state,1), round(belief,1))

    # reward function
    def reward(self, s):
        timestep, state, belief = s[0], s[1], s[2]
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
        pi = {s: None for s in self.states}
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
                            s1 = self.f(s, ah, ar1)
                            s2 = self.f(s, ah, ar2)
                            eV1 = (1-s[2]) * V[s1]
                            eV2 = s[2] * V[s2]
                            if eV1 + eV2 > v_next_max:
                                v_next_max = eV1 + eV2
                                pi[s] = [ah, ar1, ar2]
                V1[s] = self.reward(s) + v_next_max
        return pi, V1


# rollout the human and robot behavior starting at augmented state
# prints the team state and the human's belief
def rollout_team(augmented_state, pi, mdp, robot_type, human_type):
    s = copy.deepcopy(augmented_state)
    print("Belief: ", s[2], "State: ", s[1])
    for t in range(mdp.T-1):
        astar = pi[s]
        # rational human follows the optimal policy
        if human_type == "rational":
            ah = astar[0]
        # irrational human samples action at random
        # here an adversarial case occurs when human pushes right
        elif human_type == "irrational":
            ah = +0.2
        # robot follows optimal policy
        if robot_type == "confused":
            ar = astar[1]
        elif robot_type == "capable":
            ar = astar[2]
        s = mdp.f(s, ah, ar)
        print("Belief: ", s[2], "State: ", s[1])    


def main():

    # choose initial augmented state
    # (timestep t, state s, belief b)
    augmented_state = (0, 0.6, 0.2)
    if args.example == "rationally":
        augmented_state = (0, 1.0, 0.2)

    # get optimal policy for human and robot
    block1d = ExampleSBG()
    pi, V = block1d.value_iteration()

    print("[*] Confused Robot with Rational Human")
    rollout_team(augmented_state, pi, block1d, "confused", "rational")
    
    print("[*] Confused Robot with Irrational Human")
    rollout_team(augmented_state, pi, block1d, "confused", "irrational")

    print("[*] Capable Robot with Rational Human")
    rollout_team(augmented_state, pi, block1d, "capable", "rational")
    
    print("[*] Capable Robot with Irrational Human")
    rollout_team(augmented_state, pi, block1d, "capable", "irrational")

main()
