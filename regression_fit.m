function mean_dist = regression_fit(train_x, train_y, val_x, val_y, params)
  nr_inputs = size(train_x, 2);
  nr_outputs = size(train_y, 2);

  % initialize neural network
  rand('state',0)                 % use fixed random seed to make results comparable
  nn = nnsetup([nr_inputs 1000 1000 nr_outputs]);      % number of nodes in layers - input, hidden, output
  nn.learningRate = params(1);        % multiply gradient by this when changing weights
  nn.momentum = params(2);              % inertia - add this much of previous weight change
  nn.scaling_learningRate = params(3); % multiply learning rate by this after each epoch
  %nn.activation_function = 'tanh_opt';   % activation function: tanh_opt, sigm or relu
  nn.dropoutFraction = params(4);     % disable this much hidden nodes during each iteration
  nn.weightPenaltyL2 = params(5);     % penalize big weights by subtracting 
                                  % fraction of weights at each training iteration
  nn.output = 'linear';           % use linear output for regression
  opts.numepochs = 40;            % number of full sweeps through data
  opts.batchsize = 100;           % take a mean gradient step over this many samples
  opts.plot = 0;                  % enable plotting

  % early stop parameters
  opts.patience = 1;              % by default have patience for 10 epochs
  opts.patience_increase = 5;     % in case of improvement increase patience by this number * epochs
  opts.improve_threshold = 0.995; % how much new validation loss must improve over previous to be significant

  disp(nn);

  % train neural network and also plot validation error
  nn = nntrain(nn, train_x, train_y, opts, val_x, val_y);

  % predict coordinates on validation set
  val_pred = nnpredict(nn, val_x);

  % calculate mean Euclidian distance between actual and predicted coordinates
  mean_dist = mean(abs(val_y - val_pred));
  disp(['Average error: ' num2str(mean_dist) 'cm']);
end
