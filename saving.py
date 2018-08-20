'''
Created on Aug 20, 2018

@author: Lenovo
'''

import pickle
import utils as ut


def phi_loss(results):
    error = [0]*5
    for res in results:
        phi = [res.phi_n, res.phi_p, res.phi_0, res.phi_L, res.phi_S]
        for i in range(len(phi)):
            error[i] += abs(res.phi_star - phi[i]) / len(results)
    return error


def theta_loss(results):
    error = [0] * 5
    for res in results:
        theta = [res.theta_n, res.theta_p, res.theta_0, res.theta_L, res.theta_S]
        for i in range(len(theta)):
            error[i] += ut.reward_error(res.theta_star, theta[i]) / len(results)
    return error


def policy_loss(results):
    loss = [0] * 6
    for res in results:
        pi_star, _ = ut.policy_iteraion(res.mdp, res.theta_star, res.D)
        loss[0] += ut.regret(res.mdp, res.theta_star, pi_star, res.D) / len(results)
        theta = [res.theta_n, res.theta_p, res.theta_0, res.theta_L, res.theta_S]
        for i in range(len(theta)):
            pi, _ = ut.policy_iteraion(res.mdp, theta[i], pi_star)
            loss[i+1] += ut.regret(res.mdp, res.theta_star, pi_star, pi) / len(results)
    return loss


def get_user(results, user_number):
    results_filter = []
    for res in results:
        if res.ID[0] == user_number:
            results_filter.append(res)
    return results_filter


def get_iter(results, t):
    results_filter = []
    for res in results:
        if res.ID[1] == t:
            results_filter.append(res)
    return results_filter


def save_object(obj, path, append = False):
    append_write = "wb"
    if append:
        append_write = "ab"
    with open(path, append_write) as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
    
def load_results(path_pickle):
    results = []
    with open(path_pickle, 'rb') as pickle_file:
        while True:
            try:
                obj = pickle.load(pickle_file)
                results.append(obj)
            except EOFError:
                break
    return results
        

class Results:
    
    def __init__(self, ID, mdp, theta_star, phi_star, D, alpha, data):
        self.ID = ID
        self.mdp = mdp
        self.theta_star = theta_star
        self.phi_star = phi_star
        self.D = D
        self.alpha = alpha
        self.phi_n = data[0]
        self.theta_n = data[1]
        self.phi_p = data[2]
        self.theta_p = data[3]
        self.phi_0 = data[4]
        self.theta_0 = data[5]
        self.phi_L = data[6]
        self.theta_L = data[7]
        self.phi_S = data[8]
        self.theta_S = data[9]
        