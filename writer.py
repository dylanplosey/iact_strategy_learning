'''
Created on Aug 9, 2018

@author: Lenovo
'''

import saving as sv


nFeats = 4
alpha = 30
deploy = False


simulation_name = "_f" + str(nFeats) + "_a" + str(alpha)
if deploy:
    end_name = "_D.pkl"
    deploy_name = "_D_"    
else:
    end_name = ".pkl"
    deploy_name = "_"
path_pickle = r"C:\Users\Lenovo\Dropbox\5. ICRA\simulation\s" + simulation_name + end_name
results = sv.load_results(path_pickle)


path_csv_phi = r"C:\Users\Lenovo\Dropbox\5. ICRA\simulation\s" + simulation_name + deploy_name + "phi.csv"
path_csv_theta = r"C:\Users\Lenovo\Dropbox\5. ICRA\simulation\s" + simulation_name + deploy_name + "theta.csv"
path_csv_policy = r"C:\Users\Lenovo\Dropbox\5. ICRA\simulation\s" + simulation_name + deploy_name + "policy.csv"


sv.csv_writer(sv.phi_loss(results), path_csv_phi)
sv.csv_writer(sv.theta_loss(results), path_csv_theta)
sv.csv_writer(sv.policy_loss(results), path_csv_policy)
