%input a robot action, a learning strategy, the current belief, and the
%human's model of the robot policy
%output the normalizing constant for the updated belief, where the belief
%is updated using Bayesian inference

function Z = normalizer(a, psi, b, pi)

    Z = 0;
    [~,N_theta,~] = size(pi);
    for ii = 1:N_theta
        Z = Z + b(ii) * pi(a, ii, psi);
    end
    
end
