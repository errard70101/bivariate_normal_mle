function [delta, biv_se, robust_biv_se, exit_flag] = ...
    estimate_BP_MLE(starting_value, y1, y2, z1, z2, options)

if nargin == 5
    options = optimoptions(@fmincon, 'Algorithm', ...
        'trust-region-reflective', 'FiniteDifferenceType', 'central', ...
        'MaxIterations', 400, 'OptimalityTolerance', 1e-10, ...
        'SpecifyObjectiveGradient', true, 'StepTolerance', 1e-10, ...
        'FunctionTolerance', 1e-10, 'Display', 'iter', ...
        'HessianFcn', 'objective', 'CheckGradients', true);
end 

tic
lb = [-inf(length(starting_value)-1, 1); -ones(1, 1)*1];
ub = [inf(length(starting_value)-1, 1); ones(1, 1)*1];
[delta, ~, exit_flag, ~, lambda, ~, H] = ...
   fmincon(@biv_mle, starting_value, [], [], [], [], lb, ub, [], ...
   options, y1, y2, z1, z2);
disp('The estimation takes')
toc

try
    biv_cov = invChol_mex(H);
catch
    biv_cov = inv(H);
end
biv_se = sqrt(diag(biv_cov));
robust_biv_se = cal_robust_biv_se(starting_value, y1, y2, z1, z2, biv_cov);

end