addpath(genpath('DeepLearnToolbox'));

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
nn = nnsetup([61 512 16]);     
nn.learningRate = 0.1;
nn.momentum = 0.5;
%nn.scaling_learningRate = 0.99;
nn.dropoutFraction = 0.05;
%nn.weightPenaltyL2 = 1e-3; 
nn.output = 'softmax';                   %  use softmax output
opts.numepochs = 40;                     %  Number of full sweeps through data
opts.batchsize = 100;                    %  Take a mean gradient step over this many samples
opts.plot = 1;                           %  enable plotting
nn = nntrain(nn, train_x, train_y, opts, test_x, test_y); %  nntrain takes validation set as last two arguments (optionally)
er = nntest(nn, test_x, test_y)

test_preds = nnpredict(nn, test_x);
test_truth = Y(indexes((nr_train + 1):end), :);
figure;
plot_confmatrix(test_truth, test_preds);
