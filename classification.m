% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath(genpath('DeepLearnToolbox'));

% load data
load('data/features.mat');
load('data/Blocks.mat');

% calculate sizes of training and validation set
nr_samples = size(features, 1);
nr_train = 25000;               % must be divisible by batch size
nr_val = nr_samples - nr_train;

% randomly split samples into training and validation set,
% because our samples are not evenly distributed
perm = randperm(nr_samples);
train_idx = perm(1:nr_train);
val_idx = perm((nr_train + 1):(nr_train + nr_val));

train_x = features(train_idx, :);
train_y = Y(train_idx);
val_x = features(val_idx, :);
val_y = Y(val_idx);

% try linear classifier with default parameters for baseline
linclass = fitcdiscr(train_x, train_y);
val_predy = predict(linclass, val_x);
correct = (val_y == val_predy);
linear_accuracy = sum(correct) / size(correct,1)

% prepare data for DeepLearnToolbox

% normalize input to have zero mean and unit variance
[train_x, mu, sigma] = zscore(train_x);
val_x = normalize(val_x, mu, sigma);
% DeepLearnToolbox expects one-of-N coding for classes
train_y = one_of_n(train_y);
val_y = one_of_n(val_y);

% initialize neural network
rand('state',0);                % use fixed random seed to make results comparable
nn = nnsetup([61 512 16]);      % number of nodes in layers - input, hidden, output
nn.learningRate = 0.1;          % multiply gradient by this when changing weights
nn.momentum = 0.5;              % inertia - add this much of previous weight change
nn.scaling_learningRate = 0.99; % multiply learning rate by this after each epoch
%nn.activation_function = 'tanh_opt';   % activation function: tanh_opt, sigm or relu
nn.dropoutFraction = 0.05;      % disable this much hidden nodes during each iteration
%nn.weightPenaltyL2 = 0;        % penalize big weights by subtracting 
                                % fraction of weights at each training iteration
nn.output = 'softmax';          % use softmax output for classification
opts.numepochs = 40;            % number of full sweeps through data
opts.batchsize = 100;           % take a mean gradient step over this many samples
opts.plot = 1;                  % enable plotting

% train neural network and also plot validation error
nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);
% neural network final accuracy
accuracy = 1 - nntest(nn, val_x, val_y)

% plot confusion matrix for the classes
val_predy = nnpredict(nn, val_x);
val_truth = Y(val_idx);
figure;
plot_confmatrix(val_truth, val_predy);
