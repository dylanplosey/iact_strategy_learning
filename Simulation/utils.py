'''
Utility functions needed for performing each user simulation

We here define the gridworld and our modified Policy_Walk algorithm
We also include the dependent measures Reward_Error and Policy_Loss
Finally, the simulated users are defined here, including their likelihood function

'''

import random
import math
import time
import numpy as np


EPSILON = np.nextafter(0,1)


def sample_reward(mdp, D, T, eta_theta, eta_phi, alpha, theta, phi):
    """Modified Policy_Walk (originally from Ramachandran & Amir) which also
    samples phi (in addition to theta).
    Performs MCMC by sampling theta and phi for a total of T seconds, and
    returns the mean value of theta and phi sampled"""
    theta_chain, phi_chain = [theta], [phi]
    pi, V = policy_iteraion(mdp, theta)
    P = likelihood(mdp, theta, phi, V, D, alpha)
    watchdog, count = time.time(), 0
    while time.time() - watchdog < T:
        #to mitigate the effects of local minima, sample randomly sometimes!
        if random.random() < 0.95:
            theta1, phi1 = perturb_theta(theta, eta_theta), perturb_phi(phi, eta_phi)
        else:
            theta1, phi1 = sample_theta(mdp.nFeats), perturb_phi(phi, eta_phi)
            if eta_phi > 0:
                phi1 = sample_phi()
        pi1, V1 = policy_iteraion(mdp, theta1, pi)
        P1 = likelihood(mdp, theta1, phi1, V1, D, alpha)
        if random.random() < min(1, P1 / (P + EPSILON)):
            theta_chain.append(theta1)
            phi_chain.append(phi1)
            theta, phi, pi, P = theta1, phi1, pi1, P1
            count += 1
    print("I sampled " + str(count) + " times!")
    theta_mean = np.mean(np.array(theta_chain), axis=0).tolist()
    phi_mean = sum(phi_chain)/len(phi_chain)
    return theta_mean, phi_mean


def action_likelihood(mdp, theta, phi, V, state, action, alpha):
    """The (modeled) human's policy, i.e., likelihood of taking a specific action 
    given the state, theta, phi and alpha.
    Note that this returns a probability which is NOT normalized"""
    s1 = mdp.T(state, action)
    expect = V[s1] - V[state]
    exaggerate = mdp.R(s1, theta) - mdp.R(state, theta)
    action_value = expect + phi * exaggerate
    return math.exp(alpha * action_value)


def likelihood(mdp, theta, phi, V, D, alpha):
    """Return the likelihood of the policy D given theta, phi, and alpha.
    We normalize the action likelihood (over all actions) here"""
    P = 1.0
    for s in D:
        p = action_likelihood(mdp, theta, phi, V, s, D[s], alpha)
        Z = sum(action_likelihood(mdp, theta, phi, V, s, a, alpha) for a in s.actions)
        P *= (p / Z)
    return P


def simulated_human(mdp, theta, phi, alpha, noise):
    """Return the simulated user's demonstration D, which is a policy.
    A given human action is completely random with probability noise"""
    threshold = 1.0 - noise
    _, V = policy_iteraion(mdp, theta)
    D = dict([(s,0) for s in mdp.states])
    for s in mdp.states:
        if random.random() < threshold:
            q = [action_likelihood(mdp, theta, phi, V, s, a, alpha) for a in s.actions]
            P = [q[i] / sum(q) for i in range(len(q))]
            a = np.random.choice(len(q), 1, p = P)[0]
        else:
            a = random.randint(0, len(s.actions) - 1)
        D[s] = s.actions[a]
    return D


def sample_phi():
    """Generate a random value of phi"""
    return random.uniform(-1,1)


def perturb_phi(phi,eta):
    """Randomly perturb phi by step size eta"""
    lb = max(-eta/2.0, -1 - phi)
    ub = min(eta/2.0, 1 - phi)
    return phi + random.uniform(lb, ub)


def sample_theta(nFeats):
    """Generate a random value of theta"""
    return [random.uniform(-1.0/nFeats, 1.0/nFeats) for _ in range(nFeats)]


def perturb_theta(theta, eta):
    """Randomly perturb theta by step size eta"""
    nFeats = len(theta)
    theta1 = [theta[i] for i in range(nFeats)]
    for i in range(nFeats):
        lb = max(-eta/2.0, -1.0/nFeats - theta[i])
        ub = min(eta/2.0, 1.0/nFeats - theta[i])
        theta1[i] += random.uniform(lb, ub)
    return theta1


def policy_iteraion(mdp, theta, pi0 = None):
    """Policy iteration algorithm, where the policy is initialized with pi0"""
    if pi0:
        pi = dict([(s,pi0[s]) for s in mdp.states])
    else:
        pi = dict([(s,random.choice(s.actions)) for s in mdp.states])
    while True:
        V = policy_value(mdp, theta, pi)
        unchanged = True
        for s in mdp.states:
            max_reward = V[mdp.T(s,pi[s])]
            for a in s.actions:
                curr_reward = V[mdp.T(s,a)]
                if curr_reward > max_reward + 1e-5:
                    max_reward = curr_reward
                    pi[s] = a
                    unchanged = False
        if unchanged:
            return pi, V


def policy_value(mdp, theta, pi):
    """Get the value of each state given policy pi and reward parameters theta"""
    R, T, I = np.zeros(mdp.nStates), np.zeros((mdp.nStates,mdp.nStates)), np.identity(mdp.nStates)
    for s in mdp.states:
        R[mdp.get_index(s)] = mdp.R(s,theta)
        T[mdp.get_index(s), mdp.get_index(mdp.T(s,pi[s]))] = 1
    V1 = np.dot(np.linalg.inv(I - mdp.gamma * T), R)
    V = dict([(s,0) for s in mdp.states])
    for s in mdp.states:
        V[s] = V1[mdp.get_index(s)]
    return V


def regret(mdp, theta_star, pi_star, pi):
    """Return the Policy_Loss of policy pi as compared to the optimal policy pi_star,
    given the true reward parameters theta_star: |V(pi_star) - V(pi)|_1"""
    V_star = policy_value(mdp, theta_star, pi_star)
    V = policy_value(mdp, theta_star, pi)
    return sum(V_star[s] - V[s] for s in mdp.states)


def reward_error(theta_star, theta):
    """Return the Reward_Error between the true reward parameters theta_star
    and the mean estimated reward parameters theta: |theta_star - theta|_1"""
    return sum(abs(i[0] - i[1]) for i in zip(theta_star, theta))


class State:

    def __init__(self, position = None):
        """A state contains its position in the gridworld, the set of actions
        that can be taken from that state, and the feature vector for that state"""
        self.position = position
        self.actions = []
        self.features = []


class GridWorld:

    def __init__(self, nFeats = 8, nRows = 8, nCols = 8, gamma = 0.9):
        """Intitialize the gridworld, which is a list of states"""
        self.nFeats = nFeats
        self.nRows = nRows
        self.nCols = nCols
        self.nStates = self.nRows * self.nCols
        self.gamma = gamma
        self.states = []
        for y in range(self.nRows):
            for x in range(self.nCols):
                s = State((x,y))
                for _ in range(self.nFeats):
                    s.features.append(random.choice([0,1]))
                if x > 0:
                    s.actions.append((-1,0))
                if x < self.nCols - 1:
                    s.actions.append((1,0))
                if y > 0:
                    s.actions.append((0,-1))
                if y < self.nRows - 1:
                    s.actions.append((0,1))
                self.states.append(s)

    def T(self, state, action):
        """Deterministic transition function from one state to another state"""
        return self.get_state(state.position[0] + action[0], state.position[1] + action[1])

    def R(self, state, theta):
        """Return the state reward, which is a linear combination of features weighted by theta"""
        return sum(i[0] * i[1] for i in zip(state.features, theta))

    def get_state(self, pos_x, pos_y):
        """Return the state at position (pos_x, pos_y)"""
        return self.states[pos_y * (self.nCols) + pos_x]

    def get_index(self, state):
        """Return the index of a state in the gridworld list of states"""
        pos = state.position
        return pos[1] * self.nCols + pos[0]
