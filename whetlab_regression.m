% clear workspace, add DeepLearnToolbox and Whetlab to search path
clear all;
addpath(genpath('DeepLearnToolbox'));
addpath(genpath('Whetlab-Matlab-Client'));

% load data
load('data/features_500.mat');

% calculate sizes of training and validation set
nr_samples = size(feature_matrix, 1);
nr_train = 100000;              % must be divisible by batch size
nr_val = nr_samples - nr_train;

% randomly split samples into training and validation set,
% because our samples are not evenly distributed
perm = randperm(nr_samples);
train_idx = perm(1:nr_train);
val_idx = perm((nr_train + 1):(nr_train + nr_val));

train_x = feature_matrix(train_idx, 2:71);
train_y = feature_matrix(train_idx, 72:73);
val_x = feature_matrix(val_idx, 2:71);
val_y = feature_matrix(val_idx, 72:73);

% prepare data for DeepLearnToolbox

% normalize input to have zero mean and unit variance
[train_x, mu, sigma] = zscore(train_x);
val_x = normalize(val_x, mu, sigma);

% define hyperparameters we are going to optimize
parameters = {
    % use powers of 10 for learning rate - from 10^-6 to 10^-1
    struct('name', 'log10_learning_rate', 'type', 'float', 'min', -6, 'max', -1, 'size', 1),...
    % use powers of 2 for hidden nodes - from 128 to 2048
	struct('name', 'log2_hidden_nodes', 'type', 'int', 'min', 7, 'max', 11, 'size', 1),...
    % momentum from 0 to 0.99
	struct('name', 'momentum', 'type', 'float', 'min', 0.0, 'max', 0.99, 'size', 1),...
    % weight decay from 0 to 0.1
	struct('name', 'weight_decay', 'type', 'float', 'min', 0, 'max', 0.1, 'size', 1),...
    % dropout from 0 to 0.5
    struct('name', 'dropout', 'type', 'float', 'min', 0.0, 'max', 0.5, 'size', 1),...
    % number of epochs from 10 to 100
    struct('name', 'epochs', 'type', 'int', 'min', 1, 'max', 10, 'size', 1),...
    % learning rate scaling from 0.75 to 1.0
	struct('name', 'learning_rate_scale', 'type', 'float', 'min', 0.75, 'max', 1.0, 'size', 1),...
    % activation function - either tanh, sigm or relu
	struct('name', 'activation_function', 'type', 'enum', 'options', {{'tanh_opt' 'sigm' 'relu'}}, 'size', 1)...
};

% use negative distance, because Whetlab tries to maximize the outcome
outcome.name = 'negative_distance';

% start new experiment
scientist = whetlab(...
    'tambet_regression',...     % name of the experiment - MUST BE UNIQUE!!!
    'Try to find the rat!',...  % description of the experiment
    parameters,...              % hyperparameters we are going to optimize
    outcome,...                 % optimization objective
    true, ...                  % true - resume experiment, false - start new
    '871cee3e-c6b4-460c-bd30-3ce9a48838ce');  % API key, don't touch!

% try 10 experiments
for i = 1:10
    % get new suggestion from Whetlab, print it out
    job = scientist.suggest()

    % initialize neural network using parameters from Whetlab
    rand('state',0);
    nn = nnsetup([70 2^job.log2_hidden_nodes 2]);
    nn.learningRate = 10^job.log10_learning_rate;
    nn.momentum = job.momentum;
    nn.scaling_learningRate = job.learning_rate_scale;
    nn.activation_function = job.activation_function;
    nn.dropoutFraction = job.dropout;
    nn.weightPenaltyL2 = job.weight_decay;
    nn.output = 'linear';
    opts.numepochs = 10;
    opts.batchsize = 100;

    % train neural network, no plotting this time
    nn = nntrain(nn, train_x, train_y, opts);

    % predict coordinates on validation set
    nn.testing = 1;
    nn = nnff(nn, val_x, val_y);    % do a feed-forward pass in neural network
    nn.testing = 0;
    val_pred = nn.a{end};           % extract last layer node activations

    % calculate mean Euclidian distance between actual and predicted coordinates
    mean_dist = mean(sqrt(sum((val_y - val_pred).^2, 2)));
    disp(['Average distance: ' num2str(mean_dist) 'cm']);

    % send results to Whetlab
    scientist.update(job, -mean_dist);
end

% fetch and print the best hyperparameters so far
best_job = scientist.best()

% look at the results
scientist.report();
