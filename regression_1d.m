% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath('utils', genpath('DeepLearnToolbox'));

% turn off paging in Octave
more off;

% load data
load('data/TheMatrixV2.mat');
% for V3-V4
%features = features{1,1};

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

% try linear regression with default parameters for baseline

% add bias term for regression
train_x_bias = [ones(size(train_x, 1), 1) train_x];
val_x_bias = [ones(size(val_x, 1), 1) val_x];
% predict X values
coeff1 = regress(train_y(:,1), train_x_bias);
val_pred(:,1) = val_x_bias * coeff1;
% calculate mean distance between actual and predicted position
linear_mean_dist = mean(abs(val_y - val_pred));
disp(['Baseline average error: ' num2str(linear_mean_dist) 'cm']);

% prepare data for DeepLearnToolbox

% normalize input to have zero mean and unit variance
[train_x, mu, sigma] = zscore(train_x);
val_x = normalize(val_x, mu, sigma);

% find number of inputs and outputs
nr_inputs = size(train_x, 2);
nr_outputs = size(train_y, 2);

% initialize neural network
rand('state',0)                 % use fixed random seed to make results comparable
nn = nnsetup([nr_inputs 2048 2048 2048 nr_outputs]); % number of nodes in layers - input, hidden, output
nn.learningRate = 8.4338e-04;   % multiply gradient by this when changing weights
nn.momentum = 0.7462;           % inertia - add this much of previous weight change
nn.scaling_learningRate = 0.9541;   % multiply learning rate by this after each epoch
nn.activation_function = 'tanh_opt';   % activation function: tanh_opt, sigm or relu
nn.dropoutFraction = 0.0899;    % disable this much hidden nodes during each iteration
nn.weightPenaltyL2 = 4.5751e-04;% penalize big weights by subtracting 
                                % fraction of weights at each training iteration
nn.output = 'linear';           % use linear output for regression
opts.numepochs = 40;            % number of full sweeps through data
opts.batchsize = 100;           % take a mean gradient step over this many samples
opts.plot = 1;                  % enable plotting

% train neural network and also plot validation error
nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);

% predict coordinates on validation set
val_pred = nnpredict(nn, val_x);

% calculate mean Euclidian distance between actual and predicted coordinates
mean_dist = mean(abs(val_y - val_pred));
disp(['Average error: ' num2str(mean_dist) 'cm']);
