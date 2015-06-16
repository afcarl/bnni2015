% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath('utils', genpath('DeepLearnToolbox'));

% turn off paging in Octave
more off;

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

% install package nan in Octave
% in Ubuntu terminal: sudo apt-get install octave-nan
% in Windows Octave: pkg install -auto -forge nan

% load package nan in Octave (not needed if you installed with -auto)
%pkg load nan;

% try linear regression with default parameters for baseline

% add bias term for regression
train_x_bias = [ones(size(train_x, 1), 1) train_x];
val_x_bias = [ones(size(val_x, 1), 1) val_x];
% predict X values
coeff1 = regress(train_y(:,1), train_x_bias);
val_pred(:,1) = val_x_bias * coeff1;
% predict Y values
coeff2 = regress(train_y(:,2), train_x_bias);
val_pred(:,2) = val_x_bias * coeff2;
% calculate mean Euclidian distance between actual and predicted coordinates
linear_mean_dist = mean(sqrt(sum((val_y - val_pred).^2, 2)));
disp(['Baseline average error: ' num2str(linear_mean_dist) 'cm']);

% prepare data for DeepLearnToolbox

% normalize input to have zero mean and unit variance
[train_x, mu, sigma] = zscore(train_x);
val_x = normalize(val_x, mu, sigma);

% initialize neural network
rand('state',0)                 % use fixed random seed to make results comparable
nn = nnsetup([70 1024 2]);      % number of nodes in layers - input, hidden, output
nn.learningRate = 0.001;        % multiply gradient by this when changing weights
nn.momentum = 0.9;              % inertia - add this much of previous weight change
nn.scaling_learningRate = 0.99; % multiply learning rate by this after each epoch
%nn.activation_function = 'tanh_opt';   % activation function: tanh_opt, sigm or relu
%nn.dropoutFraction = 0.05;     % disable this much hidden nodes during each iteration
%nn.weightPenaltyL2 = 1e-4;     % penalize big weights by subtracting 
                                % fraction of weights at each training iteration
nn.output = 'linear';           % use linear output for regression
opts.numepochs = 10;            % number of full sweeps through data
opts.batchsize = 100;           % take a mean gradient step over this many samples
opts.plot = 1;                  % enable plotting

% train neural network and also plot validation error
nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);

% predict coordinates on validation set
val_pred = nnpredict(nn, val_x);

% calculate mean Euclidian distance between actual and predicted coordinates
mean_dist = mean(sqrt(sum((val_y - val_pred).^2, 2)));
disp(['Average error: ' num2str(mean_dist) 'cm']);

f = figure;
for i=1:20
    figure(f);
    plot(val_y(i,1), val_y(i,2), 'b*', val_pred(i,1), val_pred(i,2), 'r*', 'MarkerSize', 10);
    xlim([80 270]);
    ylim([20 220]);
    xlabel('cm');
    ylabel('cm');
    legend('actual', 'predicted', 'Location', 'northeastoutside');
    title('Rat location');
    pause(0.5);
end
