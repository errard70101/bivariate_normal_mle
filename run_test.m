clear all
clc
% This script runs some estimations to make sure that the bivariate
% probit mle procedure is correct.


% Data Description
% ========================================================================
% The test data is from Regina T. Riphahn, Achim Wambach, and 
% Andreas Million, "Incentive Effects in the Demand for Health Care: 
% A Bivariate Panel Count Data Estimation", Journal of Applied 
% Econometrics, Vol. 18, No. 4, 2003, pp. 387-405.
% ========================================================================
% Data link
% ========================================================================
% http://qed.econ.queensu.ca/jae/2003-v18.4/riphahn-wambach-million/
% ========================================================================
% I changed the name of the data file from rwm.data to rwm.dat so that
% Matlab readtable function can recognize the format.


% Load data
dta = readtable('./J Applied Econometrics/rwm.dat', ...
    'Delimiter', ' ', 'ReadVariableNames', false, ...
    'MultipleDelimsAsOne', true, 'HeaderLines', 0);

% Check if the data is correctly loaded
assert(size(dta, 1) == 27326)
assert(size(dta, 2) == 25)

% Add variable names
dta.Properties.VariableNames = {'id', 'female', 'year', 'age', 'hsat', ...
    'handdum', 'handper', 'hhninc', 'hhkids', 'educ', 'married', ...
    'haupts', 'reals', 'fachhs', 'abitur', 'univ', 'working', ...
    'bluec', 'whitec', 'self', 'beamt', 'docvis', 'hospvis', 'public', ...
    'addon'};

writetable(dta, "./J Applied Econometrics/rwm.csv")


% Generate dependent variables
dta.hospital = dta.hospvis > 0;
dta.doctor = dta.docvis > 0;
dta.const = ones(size(dta, 1), 1);

% Scale some variables
dta.hhninc = dta.hhninc/10000;

assert(sum(dta.hospital) == 2395)
assert(sum(dta.doctor) == 17191)
assert(sum(dta.hospital .* dta.doctor) == 1975)
assert(27326 - ...
    sum(dta.hospital + dta.doctor - dta.hospital .* dta.doctor) == 9715)

%% Run the test

z = [dta.const, dta.female, dta.age, dta.hhninc, dta.hhkids, dta.educ, ...
    dta.married];
[delta, biv_se, exit_flag] = ...
    estimate_BP_MLE(zeros(15, 1), dta.doctor, dta.hospital, z, z);

% Test parameter values from Greene, Econometric Analysis 7th Ed. p744,
% table 17.15
greene_est = [-0.1243; 0.3551; 0.01188; -0.1337; -0.1523; -0.01484; 0.07351;...
    -1.3385; 0.1050; 0.00461; 0.04441; -0.01517; -0.02191; -0.04789; 0.2981];
greene_se = [0.05814; 0.01604; 0.000802; 0.04628; 0.01825; 0.003575; ...
    0.02063; 0.07957; 0.02174; 0.001058; 0.05946; 0.02570; 0.005110; ...
    0.02777; 0.0139];

max_est_diff = max(abs((greene_est - delta)./greene_est));
max_se_diff = max(abs((greene_se - biv_se)./greene_se));

if max_est_diff < 1e-3
    disp('The bivariate probit MLE function works!!')
else
    disp('The test is not passed, please contact the author.')
end