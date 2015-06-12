% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath(genpath('DeepLearnToolbox'));
addpath(genpath('Whetlab-Matlab-Client'));

parameters = {struct('name', 'LR', 'type', 'float', 'min', -6, 'max', -1, 'size', 1),...
	      struct('name', 'drop', 'type', 'float', 'min', 0.0, 'max', 0.9, 'size', 1),...
       	      struct('name', 'epochs', 'type', 'enum', 'options', {{'epoch20' 'epoch40' 'epoch80'}}, 'size', 1),...
	      struct('name', 'activation', 'type', 'enum', 'options', {{'tanh_opt' 'sigm' 'relu'}}, 'size', 1),...
	      struct('name', 'hidden', 'type', 'enum', 'options', {{'size64' 'size128' 'size256' 'size512' 'size1024' 'size2048'}}, 'size', 1),...
	      struct('name', 'momentum', 'type', 'float', 'min', 0.0, 'max', 0.99, 'size', 1),...
	      struct('name', 'LR_decrease', 'type', 'float', 'min', 0.75, 'max', 1.0, 'size', 1),...
	      struct('name', 'regularization', 'type', 'float', 'min', 1e-6, 'max', 0.5, 'size', 1)};
		  
outcome.name = 'negDistance';

		  
scientist = whetlab('Whetlab_regression',...
	            'Try to find the rat!',...
	            parameters,...
	            outcome, true, '871cee3e-c6b4-460c-bd30-3ce9a48838ce');

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
clear feature_matrix

% prepare data for DeepLearnToolbox

% normalize input to have zero mean and unit variance
[train_x, mu, sigma] = zscore(train_x);
val_x = normalize(val_x, mu, sigma);

% initialize neural network
rand('state',0)                 % use fixed random seed to make results comparable


for i = 1:10
	    job = scientist.suggest()
	    size_hid=str2num(job.hidden(5:end));
	    nn = nnsetup([70 size_hid 2]);
	    nn.activation_function =job.activation;
	    nn.learningRate = 10^job.LR; %learning rate is in log scale
	    nn.momentum = job.momentum;
	    nn.dropoutFraction = job.drop;
	    nn.weightPenaltyL2 = job.regularization;
	    nn.scaling_learningRate = job.LR_decrease;
	    nn.output = 'linear';           % use linear output for regression
	    opts.numepochs = str2num(job.epochs(6:end));            % number of full sweeps through data
	    opts.batchsize = 100;           % take a mean gradient step over this many samples
	    opts.plot = 0;                  % enable plotting


	    
	    % train neural network and also plot validation error
	    nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);

	    % predict coordinates on validation set
	    nn.testing = 1;
	    nn = nnff(nn, val_x, val_y);    % do a feed-forward pass in neural network
	    nn.testing = 0;
	    val_pred = nn.a{end};           % extract last layer node activations
	    disp(size(val_pred))
	    % calculate mean Euclidian distance between actual and predicted coordinates
	    mean_dist = mean(sqrt(sum((val_y - val_pred).^2, 2)))
	    disp(['Average error: ' num2str(mean_dist) 'cm']);
	    if isnan(mean_dist) % if distance is too big to be calcluated, it's a NaN, so w just give it some big number
		mean_dist=10101010
	    end
	    scientist.update(job, -mean_dist);
end
best_job = scientist.best()
