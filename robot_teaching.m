clear
close all
clc



%select the parameters for the simulation
PSI = [1;2]; %set of learning strategies
THETA = [1;2]; %set of target models (hypotheses)
A = linspace(1,6,6)'; %set of possible robot actions
b_0_psi = [0.8;0.2]; %prior belief over learning strategies
b_0_theta = [0.5;0.5]; %prior belief over target models
n_users = 1e5; %select the number of simulated users
T = 6; %select the number of timesteps



%create pi (the human's model of the robot's policy) which outputs the 
%probability of robot action a given theta and psi
pi(:,1,1) = [0.1; 0.3; 0.45; 0.05; 0.05; 0.05];
pi(:,2,1) = [0.05; 0.05; 0.05; 0.1; 0.3; 0.45];
pi(:,1,2) = [0.35; 0.2; 0.15; 0.1; 0.1; 0.1];
pi(:,2,2) = [0.1; 0.1; 0.1; 0.35; 0.2; 0.15];



%initialize the human's actual belief in theta_star for each method
B_star = zeros(n_users,T); %star = the robot knows psi^*
B_psi_1 = zeros(n_users,T); %psi_1 = the robot assumes psi^* = psi_1
B_psi_2 = zeros(n_users,T); %psi_2 = the robot assumes psi^* = psi_2
B_0 = zeros(n_users,T);  %b_0 = the robot has a fixed belief over psi 
B = zeros(n_users,T);  %b = the robot learns about psi



%initialize the robot's prediction of the human's next state for each method
b_star_hat = zeros(length(A),1);
b_psi_1_hat = zeros(length(A),1);
b_psi_2_hat = zeros(length(A),1);
b_0_hat = zeros(length(A),1);
b_hat = zeros(length(A),1);



%perform the simulation: we test with n_users, where each user picks a
%learning strategy, and the robot samples a target model
for user_number = 1:n_users
    
    
    %initialize the expert robot, where the robot knows the target model
    theta_star = sampleFromVector(b_0_theta); %get theta_star
    b_psi = b_0_psi; %initialize belief (i.e., prior) over the human's learing strategy
    
    
    %create the simulated human: sample a learning strategy
    %note in some simulations we test the methods when the robot has the
    %wrong prior on learning strategies (just change b_0_psi here)
    psi_star = sampleFromVector(b_0_psi); %get psi_star
    
    
    %initialize the human's belief over theta for each method: this is the
    %human action at time t=0
    u_star = b_0_theta;
    u_psi_1 = b_0_theta;
    u_psi_2 = b_0_theta;
    u_0 = b_0_theta;
    u_b = b_0_theta;
    

    %now we are ready to use each method with the simulated user. The robot
    %repeats the task with the same simulated human (i.e., a human with the
    %same theta_star and phi_star) for a total of T timesteps    
    for t = 1:T
        
        
        %select the robot action that is used to teach the human;
        %first we iterate through all the actions, and record how that
        %action is expected to change the human's belief in theta_star
        for ii = 1:length(A)
            %use Bayesian inference to determine how the robot action is
            %anticipated to alter the human's belief in theta_star
            b_star_hat(ii) = u_star(theta_star) * pi(ii,theta_star,psi_star) / normalizer(ii, psi_star, u_star, pi);
            b_psi_1_hat(ii) = u_psi_1(theta_star) * pi(ii,theta_star,PSI(1)) / normalizer(ii, PSI(1), u_psi_1, pi);
            b_psi_2_hat(ii) = u_psi_2(theta_star) * pi(ii,theta_star,PSI(2)) / normalizer(ii, PSI(2), u_psi_2, pi);
            b_0_hat(ii) = u_0(theta_star) * pi(ii,theta_star,PSI(1)) / normalizer(ii, PSI(1), u_0, pi) * b_0_psi(1) + ...
                u_0(theta_star) * pi(ii,theta_star,PSI(2)) / normalizer(ii, PSI(2), u_0, pi) * b_0_psi(2);
            b_hat(ii) = u_b(theta_star) * pi(ii,theta_star,PSI(1)) / normalizer(ii, PSI(1), u_b, pi) * b_psi(1) + ...
                u_b(theta_star) * pi(ii,theta_star,PSI(2)) / normalizer(ii, PSI(2), u_b, pi) * b_psi(2); 
        end
        %find the optimal robot action by selecting the action which
        %results in the maximum belief in theta_star
        [~,a_star] = max(b_star_hat);
        [~,a_psi_1] = max(b_psi_1_hat);
        [~,a_psi_2] = max(b_psi_2_hat);
        [~,a_0] = max(b_0_hat);
        [~,a_b] = max(b_hat);
        
        
        %for the method "b", where the robot learns about psi, we need to
        %determine the predicted human belief given the robot action a_b; 
        %we get this prediction with both learning strategies
        u_hat_1 = [u_b(1)*pi(a_b,THETA(1),PSI(1)); u_b(2)*pi(a_b,THETA(2),PSI(1))] / normalizer(a_b, PSI(1), u_b, pi);
        u_hat_2 = [u_b(1)*pi(a_b,THETA(1),PSI(2)); u_b(2)*pi(a_b,THETA(2),PSI(2))] / normalizer(a_b, PSI(2), u_b, pi);
        
        
        %update the simulated human's belief over target models; this is
        %the human action at time t
        u_star = [u_star(1)*pi(a_star,THETA(1),psi_star); u_star(2)*pi(a_star,THETA(2),psi_star)] / normalizer(a_star, psi_star, u_star, pi);
        u_psi_1 = [u_psi_1(1)*pi(a_psi_1,THETA(1),psi_star); u_psi_1(2)*pi(a_psi_1,THETA(2),psi_star)] / normalizer(a_psi_1, psi_star, u_psi_1, pi);
        u_psi_2 = [u_psi_2(1)*pi(a_psi_2,THETA(1),psi_star); u_psi_2(2)*pi(a_psi_2,THETA(2),psi_star)] / normalizer(a_psi_2, psi_star, u_psi_2, pi);
        u_0 = [u_0(1)*pi(a_0,THETA(1),psi_star); u_0(2)*pi(a_0,THETA(2),psi_star)] / normalizer(a_0, psi_star, u_0, pi);
        u_b = [u_b(1)*pi(a_b,THETA(1),psi_star); u_b(2)*pi(a_b,THETA(2),psi_star)] / normalizer(a_b, psi_star, u_b, pi);
        
        
        %update the robot's belief over learning strategies using KL
        %divergence after observing the human's action
        sigma = 1e1;
        KL_1 = u_b(1)*log(u_b(1)/u_hat_1(1)) + u_b(2)*log(u_b(2)/u_hat_1(2));
        KL_2 = u_b(1)*log(u_b(1)/u_hat_2(1)) + u_b(2)*log(u_b(2)/u_hat_2(2));
        b_psi = [b_psi(1)*exp(-sigma * KL_1); b_psi(2)*exp(-sigma * KL_2)];
        b_psi = b_psi./sum(b_psi);
        
        
        %record the human's actual belief in theta_star;
        %note that the robot is aware of this belief, since it observes the
        %human's action (i.e., belief) at each timestep
        B_star(user_number,t) = u_star(theta_star);
        B_psi_1(user_number,t) = u_psi_1(theta_star);
        B_psi_2(user_number,t) = u_psi_2(theta_star);
        B_0(user_number,t) = u_0(theta_star);
        B(user_number,t) = u_b(theta_star);
            
        
        %complete the current timestep
        
    end
    
    %complete the current simulated user
    
end
