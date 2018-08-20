'''
Created on Aug 9, 2018

@author: Lenovo
'''

import random
import math
import time
import numpy as np


EPSILON = np.nextafter(0,1)


def sample_reward(mdp, D, T, eta_theta, eta_phi, alpha, theta, phi):
    theta_chain, phi_chain = [theta], [phi]
    pi, V = policy_iteraion(mdp, theta)
    P = likelihood(mdp, theta, phi, V, D, alpha)
    watchdog, count = time.time(), 0
    while time.time() - watchdog < T:
        theta1, phi1 = perturb_theta(theta, eta_theta), perturb_phi(phi, eta_phi)
        pi1, V1 = policy_iteraion(mdp, theta1, pi)
        P1 = likelihood(mdp, theta1, phi1, V1, D, alpha)
        if random.random() < min(1, P1 / (P + EPSILON)):
            theta_chain.append(theta1)
            phi_chain.append(phi1)
            theta, phi, V, pi, P = theta1, phi1, V1, pi1, P1
            count += 1
    print("I sampled " + str(count) + " times!")
    theta_mean = np.mean(np.array(theta_chain), axis=0).tolist()
    phi_mean = sum(phi_chain)/len(phi_chain)
    return theta_mean, phi_mean


def action_likelihood(mdp, theta, phi, V, state, action, alpha):
    if phi < 0:
        phi *= 0.825
    s1 = mdp.T(state, action)
    expect = V[s1] - V[state]
    exaggerate = mdp.R(s1, theta) - mdp.R(state, theta)
    action_value = expect + phi * exaggerate
    return math.exp(alpha * action_value)


def likelihood(mdp, theta, phi, V, D, alpha):
    P = 1
    for s in D:
        p = action_likelihood(mdp, theta, phi, V, s, D[s], alpha)
        Z = sum(action_likelihood(mdp, theta, phi, V, s, a, alpha) for a in s.actions)
        P *= (p / Z)
    return P


def simulated_human(mdp, theta, phi, alpha):
    _, V = policy_iteraion(mdp, theta)
    D = dict([(s,0) for s in mdp.states])
    for s in mdp.states:
        q = [action_likelihood(mdp, theta, phi, V, s, a, alpha) for a in s.actions]
        P = [q[i] / sum(q) for i in range(len(q))]
        a = np.random.choice(len(q), 1, p = P)[0]
        D[s] = s.actions[a]
    return D
    

def sample_phi():
    return random.uniform(-1,1)


def perturb_phi(phi,eta):
    lb = max(-eta/2, -1 - phi)
    ub = min(eta/2, 1 - phi)
    return phi + random.uniform(lb, ub)
    
    
def sample_theta(nFeats):
    return [random.uniform(-1/nFeats, 1/nFeats) for _ in range(nFeats)]

                    
def perturb_theta(theta, eta):
    nFeats = len(theta)
    theta1 = [theta[i] for i in range(nFeats)]
    for i in range(nFeats):
        lb = max(-eta/2, -1/nFeats - theta[i])
        ub = min(eta/2, 1/nFeats - theta[i])
        theta1[i] += random.uniform(lb, ub)
    return theta1


def policy_iteraion(mdp, theta, pi0 = None):
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
    V_star = policy_value(mdp, theta_star, pi_star)
    V = policy_value(mdp, theta_star, pi)
    return sum(V_star[s] - V[s] for s in mdp.states)


def reward_error(theta_star, theta):
    return sum(abs(i[0] - i[1]) for i in zip(theta_star, theta))


class State:

    def __init__(self, position = None):
        self.position = position
        self.actions = []
        self.features = []
        
        
class GridWorld:
    
    def __init__(self, nFeats = 8, nRows = 8, nCols = 8, gamma = 0.9):
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
        return self.get_state(state.position[0] + action[0], state.position[1] + action[1])
    
    def R(self, state, theta):
        return sum(i[0] * i[1] for i in zip(state.features, theta))
    
    def get_state(self, pos_x, pos_y):
        return self.states[pos_y * (self.nCols) + pos_x]
    
    def get_index(self, state):
        pos = state.position
        return pos[1] * self.nCols + pos[0]
    