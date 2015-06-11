% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath(genpath('DeepLearnToolbox'));

% load data
load('data/features_new.mat');

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

% try linear regression with default parameters for baseline

% predict X values
linregx = fitlm(train_x, train_y(:,1));
val_pred(:,1) = predict(linregx, val_x);
% predict Y values
linregy = fitlm(train_x, train_y(:,2));
val_pred(:,2) = predict(linregy, val_x);
% calculate mean Euclidian distance between actual and predicted coordinates
linear_mean_dist = mean(sqrt(sum((val_y - val_pred).^2, 2)));
disp(['Baseline average distance: ' num2str(round(linear_mean_dist, 2)) 'cm']);

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
nn.testing = 1;
nn = nnff(nn, val_x, val_y);    % do a feed-forward pass in neural network
nn.testing = 0;
val_pred = nn.a{end};           % extract last layer node activations

% calculate mean Euclidian distance between actual and predicted coordinates
mean_dist = mean(sqrt(sum((val_y - val_pred).^2, 2)));
disp(['Average error: ' num2str(round(mean_dist, 2)) 'cm']);

%val_dist_zhurab = sqrt(mean(sum((val_y - val_pred).^2, 2)))
%val_dist_tambet = mean(sqrt(sum((val_y - val_pred).^2, 2)))
%val_dists_zhurab = sqrt(mean((val_y - val_pred).^2, 1))
%val_dists_tambet = mean(abs(val_y - val_pred), 1)

f = figure;
for i=1:100
    figure(f);
    plot(val_y(i,1), val_y(i,2), 'b*', val_pred(i,1), val_pred(i,2), 'r*', 'MarkerSize', 10);
    xlim([80 270]);
    ylim([20 220]);
    xlabel('cm');
    ylabel('cm');
    legend('actual', 'predicted', 'Location', 'bestoutside');
    title('Rat location');
    pause(0.5);
end
