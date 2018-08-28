clear
close all
clc



%select the parameters for the simulation
n_screws = 10; %number of screws to be sorted
PHI = [1;2]; %set of teaching strategies
THETA = linspace(1,n_screws,n_screws)'; %set of target models (hypotheses)
U = linspace(1,n_screws,n_screws)'; %set of possible human (and robot) actions
b_0_phi = [0.5;0.5]; %prior belief over teaching strategies
b_0_theta = ones(n_screws,1)/n_screws; %prior belief over target models
n_users = 1e5; %select the number of simulated users
T = 100; %select the number of timesteps



%create pi (the human's policy) which outputs the probability of action u
%given theta and phi
pi = zeros(length(U),length(THETA),length(PHI)); %initialize pi
beta = 0.5; %boltzmann constant for phi_1
for ii = 1:length(THETA)
    for jj = 1:length(U)
        %for phi_1, the human noisily selects the screw closest to the
        %threshold theta using the maximum entropy distribution
        pi(jj,ii,1) = exp(-beta*abs(U(jj) - THETA(ii)));
        %for phi_2, the human selects a screw uniformly at random from the
        %set of screws less than or equal to the threshold theta
        if U(jj) <= THETA(ii)
            pi(jj,ii,2) = 0.9/THETA(ii);
        else
            pi(jj,ii,2) = 0.1/(n_screws - THETA(ii));
        end
    end
    %normalize the probability distributions
    pi(:,ii,1) = pi(:,ii,1)./sum(pi(:,ii,1));
    pi(:,ii,2) = pi(:,ii,2)./sum(pi(:,ii,2));
end



%initialize the number of screws classified incorrectly with each method
error_star = zeros(n_users,T); %star = the robot knows phi^*
error_phi_1 = zeros(n_users,T); %phi_1 = the robot assumes phi^* = phi_1
error_phi_2 = zeros(n_users,T); %phi_2 = the robot assumes phi^* = phi_2
error_0 = zeros(n_users,T);  %b_0 = the robot has a fixed belief over phi
error_b = zeros(n_users,T); %b = the robot learns theta and phi



%perform the simulation: we test with n_users, where each user picks a
%teaching strategy and desired threshold
for user_number = 1:n_users
    
    
    %create the simulated user: sample a teaching strategy and theta
    %note in some simulations we test the methods when the robot has the
    %wrong prior on teaching strategies (just change b_0_phi here)
    phi_star = sampleFromVector([0.9;0.1]); %get phi_star
    theta_star = sampleFromVector(b_0_theta);%get theta_star
    
    
    %initialize the robot's belief (i.e., the prior) for each method
    b_star = b_0_theta;
    b_phi_1 = b_0_theta;
    b_phi_2 = b_0_theta;
    b_0 = b_0_theta;
    b = [b_0_theta,b_0_theta]*diag(b_0_phi);
    
    
    %now we are ready to use each method with the simulated user. The robot
    %repeats the task with the same simulated human (i.e., a human with the
    %same theta_star and phi_star) for a total of T timesteps
    for t = 1:T
        
        
        %get the action of the simulated human by sampling from pi
        u = sampleFromVector(pi(:,theta_star,phi_star));


        %the robot observes this action, and then updates its belief over
        %theta using each of the methods
        for ii = 1:length(THETA)
            b_star(ii) = b_star(ii)*pi(u,ii,phi_star);
            b_phi_1(ii) = b_phi_1(ii)*pi(u,ii,PHI(1));
            b_phi_2(ii) = b_phi_2(ii)*pi(u,ii,PHI(2));
            b_0(ii) = b_0(ii)*(pi(u,ii,PHI(1))*b_0_phi(1) + pi(u,ii,PHI(2))*b_0_phi(2));
            for jj = 1:length(PHI)
                b(ii,jj) = b(ii,jj)*pi(u,ii,jj);
            end
        end
        
        
        %normalize these updated beliefs
        b_star = b_star./sum(b_star);
        b_phi_1 = b_phi_1./sum(b_phi_1);
        b_phi_2 = b_phi_2./sum(b_phi_2);
        b = b./sum(b(:));
        b_theta = sum(b,2);
    

        %using the updated beliefs for each method, find the optimal robot 
        %action. This is done by finding the optimal action for the mean
        %reward function.
        %start by initializing the reward for each action
        R_star = zeros(size(U));
        R_phi_1 = zeros(size(U));
        R_phi_2 = zeros(size(U));
        R_0 = zeros(size(U));
        R_b = zeros(size(U));
        %next, go through each action, and see what reward we would expect
        %to get for performing this action
        for ii = 1:length(U)
            for jj = 1:length(THETA)
                R_star(ii) = R_star(ii) + (n_screws - abs(U(ii) - THETA(jj)))*b_star(jj);
                R_phi_1(ii) = R_phi_1(ii) + (n_screws - abs(U(ii) - THETA(jj)))*b_phi_1(jj);
                R_phi_2(ii) = R_phi_2(ii) + (n_screws - abs(U(ii) - THETA(jj)))*b_phi_2(jj);
                R_0(ii) = R_0(ii) + (n_screws - abs(U(ii) - THETA(jj)))*b_0(jj);
                R_b(ii) = R_b(ii) + (n_screws - abs(U(ii) - THETA(jj)))*b_theta(jj);
            end
        end
        %finally, we select the action which maximizes the expected reward
        [~,a_star] = max(R_star);
        [~,a_phi_1] = max(R_phi_1);
        [~,a_phi_2] = max(R_phi_2);
        [~,a_0] = max(R_0);
        [~,a_b] = max(R_b);
        
    
        %we record the number of screws that were incorrectly classified
        %when using the robot's optimal action for each method
        %note that the robot is not aware of this error; this result is
        %only for us to compare the results!
        error_star(user_number,t) = abs(a_star - theta_star);
        error_phi_1(user_number,t) = abs(a_phi_1 - theta_star);
        error_phi_2(user_number,t) = abs(a_phi_2 - theta_star);
        error_0(user_number,t) = abs(a_0 - theta_star);
        error_b(user_number,t) = abs(a_b - theta_star);
        
        
        %complete the current timestep
        
    end
    
    %complete the current simulated user

end
