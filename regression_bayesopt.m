% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath('utils', genpath('DeepLearnToolbox'), genpath('bayesopt'));

% turn off paging in Octave
more off;

% load data
load('data/TheMatrixV2.mat');

% calculate sizes of training and validation set
nr_samples = size(features, 1);
nr_train = 40000;              % must be divisible by batch size
nr_val = nr_samples - nr_train;

% randomly split samples into training and validation set,
% because our samples are not evenly distributed
perm = randperm(nr_samples);
train_idx = perm(1:nr_train);
val_idx = perm((nr_train + 1):(nr_train + nr_val));

train_x = features(train_idx, :);
train_y = position(train_idx, :);
val_x = features(val_idx, :);
val_y = position(val_idx, :);

% prepare data for DeepLearnToolbox

% normalize input to have zero mean and unit variance
[train_x, mu, sigma] = zscore(train_x);
val_x = normalize(val_x, mu, sigma);

F = @(params) regression_fit(train_x, train_y, val_x, val_y, params);

% parameters for bayesopt
params = [ ...
    0.00001  0.001; ... % learning rate
    0.9 1; ...          % momentum
    0.9 1; ...          % learning rate scaling
    0 0.5; ...          % dropout
    0 10^-4;            % weight decay
];

opt = defaultopt;
opt.dims = 5;
opt.mins = params(:,1)';
opt.maxes = params(:,2)';
opt.maxiters = 25;
opt.save_trace = 1;
opt.resume_trace = 1;
opt.trace_file = 'ratgps.mat';
%opt.parallel_jobs = 2;
%parpool('local', 2);
[ min_sample, min_value, botrace ] = bayesopt(F, opt)
