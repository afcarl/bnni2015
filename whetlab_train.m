clear all;
addpath(genpath('DeepLearnToolbox'));
addpath(genpath('Whetlab-Matlab-Client'));

parameters = {struct('name', 'LR', 'type', 'float', 'min', 1e-6, 'max', 2.0, 'size', 1),...
          struct('name', 'drop', 'type', 'float', 'min', 0.0, 'max', 0.9, 'size', 1),...
          struct('name', 'activation', 'type', 'enum', 'options', {{'tanh_opt' 'sigm' 'relu'}}, 'size', 1),...
          struct('name', 'hidden', 'type', 'enum', 'options', {{'size64' 'size128' 'size256','size512','size1024','size2048'}}, 'size', 1),...
          struct('name', 'momentum', 'type', 'float', 'min', 0.0, 'max', 0.99, 'size', 1),...
          struct('name', 'LR_decrease', 'type', 'float', 'min', 0.75, 'max', 1.0, 'size', 1),...
          struct('name', 'regularization', 'type', 'float', 'min', 1e-6, 'max', 0.5, 'size', 1)};

outcome.name = 'Accuracy';


scientist = whetlab('Whetlab_search_rat',...
                'Try to find the rat!',...
                parameters,...
                outcome, true, '871cee3e-c6b4-460c-bd30-3ce9a48838ce');

load('data/features.mat');
load('data/Blocks.mat');
nr_samples = size(features, 1);
nr_train = 25000;
indexes = randperm(nr_samples);
Y2 = one_of_n(Y);

train_x = features(indexes(1:nr_train), :);
train_y = Y2(indexes(1:nr_train), :);
test_x = features(indexes((nr_train + 1):end), :);
test_y = Y2(indexes((nr_train + 1):end), :);

[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

rand('state',0)

for i = 1:10
    job = scientist.suggest();
    size_str=job.hidden
    size_hid = str2num(size_str(5:end))
    nn = nnsetup([61 size_hid 16]);
    nn.activation_function = 'relu'; %job.activation;
    nn.learningRate = job.LR;
    nn.momentum = job.momentum;
    nn.dropoutFraction = job.drop;
    nn.weightPenaltyL2 = job.regularization;
    nn.scaling_learningRate = job.LR_decrease;

    nn.output = 'softmax';                   %  use softmax output
    opts=struct;
    opts.numepochs = 10;                     %  Number of full sweeps through data
    opts.batchsize = 100;                    %  Take a mean gradient step over this many samples
    opts.plot = 0;                           %  enable plotting
    nn = nntrain(nn, train_x, train_y, opts, test_x, test_y); %  nntrain takes validation set as last two arguments (optionally)
    er = nntest(nn, test_x, test_y)
    disp(1-er)
    scientist.update(job,1-er);
end
best_job = scientist.best()
