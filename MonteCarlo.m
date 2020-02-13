clc
clear all

% This script runs Monte Carlo experiments to test the performance of the
% code.


% Setup Parameters
%==========================================================================
n_obs = 500;
n_sim = 5000;
beta = [1; -1];
MU = [0, 0];
rho = 0.5;
Sigma = [1, rho; rho, 1];
true_params = [beta;beta;rho];

n_params = size(true_params, 1);
%==========================================================================

% Create matrices to store simulation results
estimates = zeros(n_sim, n_params);

test_result = zeros(size(true_params, 1), n_sim);
robust_test_result = zeros(size(true_params, 1), n_sim);

i = 1;
while i <= n_sim
    
    if mod(i, 100) == 0
        disp(['This is the ', num2str(i), 'th simulation.'])
    end
    
% Generate data
e = mvnrnd(MU, Sigma, n_obs);
const = ones(n_obs, 1);
X1 = (rand(n_obs, size(beta, 1) - 1) - 0.5 )*5;
X2 = (rand(n_obs, size(beta, 1) - 1) - 0.5 )*5;
y1 = [const, X1]*beta + e(:, 1) > 0;
y2 = [const, X2]*beta + e(:, 2) > 0;

% Estimate the model
starting_value = zeros(n_params, 1);

options = optimoptions(@fmincon, 'Algorithm', ...
        'trust-region-reflective', 'FiniteDifferenceType', 'central', ...
        'MaxIterations', 400, 'OptimalityTolerance', 1e-10, ...
        'SpecifyObjectiveGradient', true, 'StepTolerance', 1e-10, ...
        'FunctionTolerance', 1e-10, 'Display', 'off', ...
        'HessianFcn', 'objective', 'CheckGradients', false);
    
lb = [-inf(length(starting_value)-1, 1); -1];
ub = [inf(length(starting_value)-1, 1); 1];

% Estimation procedure
try
    [delta, ~, exit_flag, ~, lambda, ~, H] = ...
        fmincon(@biv_mle, starting_value, [], [], [], [], lb, ub, [], ...
        options, y1, y2, [const, X1], [const, X2]);
catch
    disp('Estimation fails.')
end

biv_se = sqrt(diag(inv(H)));
robust_biv_se = cal_robust_biv_se(delta, y1, y2, ...
    [const, X1], [const, X2]);

test = abs((delta - true_params)./biv_se) > 1.96;
robust_test = abs((delta - true_params)./robust_biv_se) > 1.96;

estimates(i, :) = delta';

test_result(:, i) = test;
robust_test_result(:, i) = robust_test;

i = i + 1;
end

bias = (mean(estimates)' - true_params)';
rmse = sqrt(mean((estimates - true_params').^2));

rej_rate = (sum(test_result, 2)./n_sim.*100)';
robust_rej_rate = (sum(robust_test_result, 2)./n_sim.*100)';