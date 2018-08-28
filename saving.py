'''
Created on Aug 20, 2018

@author: Lenovo
'''

import pickle
import csv
import utils as ut
import numpy as np


def phi_loss(results):
    loss = []
    for res in results:
        phi = [res.phi_n, res.phi_p, res.phi_0, res.phi_L, res.phi_S]
        loss.append(phi)
    return loss


def theta_loss(results):
    loss = np.zeros((len(results),5))
    error = [0] * 5
    for res in results:
        theta = [res.theta_n, res.theta_p, res.theta_0, res.theta_L, res.theta_S]
        for i in range(len(theta)):
            error[i] = ut.reward_error(res.theta_star, theta[i])
        loss[res.ID,:] = error
    return loss.tolist()


def policy_loss(results):
    loss = np.zeros((len(results),5))
    error = [0] * 5
    for res in results:
        pi_star, _ = ut.policy_iteraion(res.mdp, res.theta_star, res.D)
        theta = [res.theta_n, res.theta_p, res.theta_0, res.theta_L, res.theta_S]
        for i in range(len(theta)):
            pi, _ = ut.policy_iteraion(res.mdp, theta[i], pi_star)
            error[i] = ut.regret(res.mdp, res.theta_star, pi_star, pi)
        loss[res.ID,:] = error
    return loss.tolist()
        
    
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


def csv_writer(data, path, append = False):
    append_write = "w"
    if append:
        append_write = "a"
    with open(path, append_write, newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def save_object(obj, path, append = False):
    append_write = "wb"
    if append:
        append_write = "ab"
    with open(path, append_write) as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        

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
        