% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath('utils', genpath('DeepLearnToolbox'), genpath('bayesopt'));

% turn off paging in Octave
more off;

% load data
load('data/TheMatrixV4.mat');
features = features{1,2};

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
    -5  -3; ...         % learning rate in log10 scale
    0.5 1; ...          % momentum
    0.9 1; ...          % learning rate scaling
    0 0.5; ...          % dropout
    -5 -2; ...          % weight decay in log10 scale
    8 12; ...           % nr of hidden nodes in log2 scale
    1 3;                % nr of hidden layers
];

opt = defaultopt;
opt.dims = 7;
opt.mins = params(:,1)';
opt.maxes = params(:,2)';
opt.maxiters = 25;
opt.save_trace = 1;
opt.resume_trace = 1;
opt.trace_file = 'ratgpsV4b.mat';
%opt.parallel_jobs = 2;
%parpool('local', 2);
[ min_sample, min_value, botrace ] = bayesopt(F, opt)
