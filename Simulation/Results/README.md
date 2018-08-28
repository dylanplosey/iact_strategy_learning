# Learning Simulation Results

The results used to generate the figures from Section VI-E. In other words, the results from our Learning Simulations.

## Nomenclature
The files are named as follows:

- fXX gives the number of features for that simulation: |F| = 4, 8, and 16
- aXX gives the value of alpha for that simulation: alpha = 5, 10, and 20
- nXX.X gives the ratio of noised used for that simulation. We only included these in the name when noise was introduced (i.e., when |F| = 8 and alpha = 10). The values used are noise = 10.0, 20.0, and 40.0.

## What's in the Files?
The files can be used to obtain the Policy_Loss (policy), Reward_Error (theta), and Strategy_Error (phi) for each individual user and each different condition.

- .pkl : these files are a list of Results objects (see "saving.py"), which include the gridworld, demonstration, theta_star, phi_star, and what was learned by each type of robot
- .csv : these files include the Policy_Loss (policy), Reward_Error (theta), and the learned value of phi_star (phi). Each row is a different simulated user, and the columns are as follows: [phi = -1, phi = +1, phi = 0, Joint, phi^*]
- .m : these files are simply the .csv files, but prepared for use in MATLAB. When imported to MATLAB, you will get matrices for the Policy_Loss (policy), Reward_Error (theta), and the learned value of phi_star (phi)
