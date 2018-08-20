'''
Created on Aug 9, 2018

@author: Lenovo

Notes:

1. need to be able to save the results...seems fine, using pickle
2. run on other computers...seems fine, github would be nice
3. read in the results and interpret for plotting...read in correctly

confirm that results are what we expect

4. performance not as good as the number of features increases...need to calibrate range
    this can wait till tomorrow

'''


import utils as ut
import saving as sv
import math


nFeats = 4
alpha = 20
L = 5
n_users = 11
T = 4 


simulation_name = "_f" + str(nFeats) + "_a" + str(alpha)
path_pickle = r"C:\Users\Lenovo\Dropbox\5. ICRA\simulation\s" + simulation_name + ".pkl"
append = False


for user_number in range(n_users):
    
    theta_star, theta0 = ut.sample_theta(nFeats), ut.sample_theta(nFeats)
    theta0_n, theta0_p, theta0_0, theta0_L, theta0_S = theta0, theta0, theta0, theta0, theta0
    phi_star = -1 + float(user_number)/(n_users-1.0) * 2
    phi0 = ut.sample_phi() 
    
    for t in range(T):
        
        print("User " + str(user_number+1) + " of " + str(n_users) + " on Iteration " + str(t+1) + " of " + str(T))

        mdp = ut.GridWorld(nFeats)
        D = ut.simulated_human(mdp, theta_star, phi_star, alpha)
    
        eta_theta = 0.5/nFeats# * math.exp(-t)
        eta_phi = 1.0 * math.exp(-t)
        theta_mean_n, phi_mean_n = ut.sample_reward(mdp, D, L, eta_theta, 0.0, alpha, theta0_n, -1.0)
        theta_mean_p, phi_mean_p = ut.sample_reward(mdp, D, L, eta_theta, 0.0, alpha, theta0_p, 1.0)
        theta_mean_0, phi_mean_0 = ut.sample_reward(mdp, D, L, eta_theta, 0.0, alpha, theta0_0, 0.0)
        theta_mean_L, phi_mean_L = ut.sample_reward(mdp, D, L, eta_theta, eta_phi, alpha, theta0_L, phi0)
        theta_mean_S, phi_mean_S = ut.sample_reward(mdp, D, L, eta_theta, 0.0, alpha, theta0_S, phi_star)

        theta0_n = theta_mean_n
        theta0_p = theta_mean_p
        theta0_0 = theta_mean_0
        theta0_L = theta_mean_L
        theta0_S = theta_mean_S        
        phi0 = phi_mean_L
        
        data = [phi_mean_n, theta_mean_n, phi_mean_p, theta_mean_p, phi_mean_0, theta_mean_0, 
                phi_mean_L, theta_mean_L, phi_mean_S, theta_mean_S]
        obj = sv.Results((user_number,t), mdp, theta_star, phi_star, D, alpha, data)
        sv.save_object(obj, path_pickle, append)
        append = True

