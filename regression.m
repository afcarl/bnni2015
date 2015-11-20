% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath('utils', genpath('DeepLearnToolbox'));

% turn off paging in Octave
more off;

% load data
load('data/TestMatrix40ms3win0.5slide10speed.mat');
spikes = features;
coords = position;

% calculate sizes of training and validation set
nr_samples = size(spikes, 1);
nr_train = round(nr_samples * 0.8, -2);              % must be divisible by batch size
nr_val = nr_samples - nr_train;
nr_neurons = size(spikes, 2);
nr_coords = size(coords, 2);

% randomly split samples into training and validation set,
% because our samples are not evenly distributed
train_idx = randperm(nr_train);
val_idx = nr_train + randperm(nr_val);

train_x = spikes(train_idx, :);
train_y = coords(train_idx, :);
val_x = spikes(val_idx, :);
val_y = coords(val_idx, :);

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
nn = nnsetup([nr_neurons 1024 1024 nr_coords]);      % number of nodes in layers - input, hidden, output
nn.learningRate = 0.001;        % multiply gradient by this when changing weights
nn.momentum = 0.9;              % inertia - add this much of previous weight change
nn.scaling_learningRate = 0.99; % multiply learning rate by this after each epoch
%nn.activation_function = 'tanh_opt';   % activation function: tanh_opt, sigm or relu
nn.dropoutFraction = 0.1;     % disable this much hidden nodes during each iteration
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

% compare actual and predicted mouse locations
sim_idx = 1:10:1000;              % dataset slice to use
sim_x = spikes(sim_idx, :);       % spikes from original dataset
sim_y = coords(sim_idx, :);       % actual coordinates from original dataset
sim_pred = nnpredict(nn, sim_x);  % predicted coordinates from spikes
f = figure;
for i=1:size(sim_x,1)
    figure(f);
    plot(sim_y(i,1), sim_y(i,2), 'b*', sim_pred(i,1), sim_pred(i,2), 'r*', 'MarkerSize', 10);
    xlim([80 270]);
    ylim([20 220]);
    xlabel('cm');
    ylabel('cm');
    legend('actual', 'predicted', 'Location', 'northeastoutside');
    title('Rat location');
    drawnow;
end
