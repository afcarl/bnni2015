% clear workspace and add DeepLearnToolbox to search path
clear all;
addpath(genpath('DeepLearnToolbox'));
addpath(genpath('Whetlab-Matlab-Client'));

parameters = {struct('name', 'LR', 'type', 'float', 'min', 1e-6, 'max', 2.0, 'size', 1),...
	      struct('name', 'drop', 'type', 'float', 'min', 0.0, 'max', 0.9, 'size', 1),...
              struct('name', 'epochs', 'type', 'int', 'min', 10, 'max', 200, 'size', 1),...
	      struct('name', 'activation', 'type', 'enum', 'options', {{'tanh_opt' 'sigm' 'relu'}}, 'size', 1),...
	      struct('name', 'hidden', 'type', 'int', 'min', 32, 'max', 2048, 'size', 1),...
	      struct('name', 'momentum', 'type', 'float', 'min', 0.0, 'max', 0.99, 'size', 1),...
	      struct('name', 'LR_decrease', 'type', 'float', 'min', 0.75, 'max', 1.0, 'size', 1),...
	      struct('name', 'regularization', 'type', 'float', 'min', 1e-6, 'max', 0.5, 'size', 1)};
		  
outcome.name = 'negDistance';

		  
scientist = whetlab('Whetlab_rat_regression',...
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

for LR= [0.00001 0.001]
	job.LR=LR;
	for hidd=[64 512]
		job.hidden=hidd;
		for momentum=[0.05 0.75]
			job.momentum=momentum;
			for LR_decrease=[0.95 1.0]
				job.LR_decrease=LR_decrease;
				for drop=[0.05, 0.5]
					job.drop=drop;
					for activation={'sigm' 'relu' 'tanh_opt'}
						job.activation=activation{1};
						job.regularization=1e-6;
						job.epochs=20
						nn = nnsetup([70 job.hidden 2]);
	    					nn.activation_function =job.activation;
						nn.learningRate = job.LR;
						nn.momentum = job.momentum;
						nn.dropoutFraction = job.drop;
						nn.weightPenaltyL2 = job.regularization;
						nn.scaling_learningRate = job.LR_decrease;
						nn.output = 'linear';           % use linear output for regression
						opts.numepochs = job.epochs;            % number of full sweeps through data
						opts.batchsize = 100;           % take a mean gradient step over this many samples
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
						if isnan(mean_dist)
							mean_dist=10101010
						end
						scientist.update(job, -mean_dist);
					end
				end
			end
		end
	end
end


best_job = scientist.best()
