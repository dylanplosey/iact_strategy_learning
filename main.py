'''
Main file to run the simulations from Section "Learning Simulations"

You can set the parameters for the simulation, including where the results will be saved
Running this code stores a Pickle containing each simulated user's results at "path_pickle"

'''

import utils as ut
import saving as sv


nFeats = 8 		#number of features at each state: tested |F| = 4, 8, and 16
alpha = 10 		#human's rationality: tested alpha = 5, 10, and 20
noise = 0.0 		#ratio of additional noise in simulation: tested noise = 0, 10, 20, and 40
T = 5 			#amount of time that the robot can sample theta (and phi) using Policy_Walk
n_users = 100 		#number of simulated users for each trial
eta_theta = 0.5/nFeats 	#step size for perturbing theta in Policy_Walk
eta_phi = 1.0 		#step size for perturbing phi in Policy_Walk


#select what to call and where to store the simulation results
simulation_name = "_f" + str(nFeats) + "_a" + str(alpha) + "_n" + str(noise*100)
path_pickle = r"C:\Users\MAHI\Desktop\Dylan's Python Code\s" + simulation_name + ".pkl"
append = False


#for each user, get the user's demonstration, and then learn with each type of robot
for user_number in range(n_users):
	
    #randomly generate a value of theta^* for the user
    theta_star, theta0 = ut.sample_theta(nFeats), ut.sample_theta(nFeats)
    #randomly generate a value of phi^* for the user
    phi_star, phi0 = -1.0 + float(user_number)/(n_users-1.0) * 2.0, 0.0

    print("User " + str(user_number+1) + " of " + str(n_users))

    #randomly generate a gridworld mdp (8x8) with randomly initialized features
    mdp = ut.GridWorld(nFeats)
    #get the human demonstration as a function of theta^*, phi^*, alpha, and the ratio of noise
    D = ut.simulated_human(mdp, theta_star, phi_star, alpha, noise)

    #sample theta and phi with each type of robot, and obtain the mean value for each
    #note that we are using a modified version of Policy_Walk here
    theta_mean_n, phi_mean_n = ut.sample_reward(mdp, D, T, eta_theta, 0.0, alpha, theta0, -1.0)		#learner with phi = -1
    theta_mean_p, phi_mean_p = ut.sample_reward(mdp, D, T, eta_theta, 0.0, alpha, theta0, 1.0)		#learner with phi = +1
    theta_mean_0, phi_mean_0 = ut.sample_reward(mdp, D, T, eta_theta, 0.0, alpha, theta0, 0.0)		#learner with phi = 0 (Prior)
    theta_mean_L, phi_mean_L = ut.sample_reward(mdp, D, T, eta_theta, eta_phi, alpha, theta0, phi0)	#Joint learner
    theta_mean_S, phi_mean_S = ut.sample_reward(mdp, D, T, eta_theta, 0.0, alpha, theta0, phi_star)	#ideal learner

    #save the results for the current user to the Pickle file
    data = [phi_mean_n, theta_mean_n, phi_mean_p, theta_mean_p, phi_mean_0, theta_mean_0,
            phi_mean_L, theta_mean_L, phi_mean_S, theta_mean_S]
    obj = sv.Results(user_number, mdp, theta_star, phi_star, D, alpha, data)
    sv.save_object(obj, path_pickle, append)
    append = True
