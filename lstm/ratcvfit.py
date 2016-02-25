import numpy as np
import argparse
import math
import os.path
from ratdata import *
from ratlstm import *
from ratbilstm import *

def cvfit(model, X, y, args):
  n = X.shape[0]
  foldsize = math.ceil(n / float(args.cvfolds))

  # remember initial weights
  weights = model.get_weights()
  results = []
  for i in xrange(args.cvfolds):
    test_X = X[i*foldsize:(i+1)*foldsize]
    test_y = y[i*foldsize:(i+1)*foldsize]
    rest_X = np.vstack((X[:i*foldsize], X[(i+1)*foldsize:]))
    rest_y = np.vstack((y[:i*foldsize], y[(i+1)*foldsize:]))
    print "test: [%d:%d], rest: [:%d], [%d:]" % (i*foldsize, (i+1)*foldsize, i*foldsize, (i+1)*foldsize)
    print test_X.shape, test_y.shape, rest_X.shape, rest_y.shape

    if args.train_set == 1:
      train_X = rest_X
      train_y = rest_y
      valid_X = test_X
      valid_y = test_y
    else:
      train_X, train_y, valid_X, valid_y = split_data(rest_X, rest_y, args.train_set, args.split_shuffle)

    model_path = args.save_path + "-" + str(i + 1) + ".hdf5"
    model.set_weights(weights)
    model.fit(train_X, train_y, valid_X, valid_y, model_path)
    train_err, train_dist = model.eval(train_X, train_y, model_path)
    valid_err, valid_dist = model.eval(valid_X, valid_y, model_path)
    test_err, test_dist = model.eval(test_X, test_y, model_path)
    print 'train mse = %g, validation mse = %g, test mse = %g' % (train_err, valid_err, test_err)
    print 'train dist = %g, validation dist = %g, test dist = %g' % (train_dist, valid_dist, test_dist)
    results.append((train_dist, valid_dist, test_dist))

  mean_dist = tuple(np.mean(results, axis=0))
  print "train mean dist = %g, valid mean dist = %g, test mean dist = %g" % mean_dist
  return mean_dist

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_data_params(parser)
  add_model_params(parser)
  parser.add_argument("save_path")
  parser.add_argument("--cvfolds", type=int, default=5)
  parser.add_argument("--model", choices=['lstm', 'bilstm'], default='lstm')
  args = parser.parse_args()

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args.seqlen)

  if args.model == 'lstm':
    model = RatLSTM(**vars(args))
  elif args.model == 'bilstm':
    model = RatBiLSTM(**vars(args))
  else:
    assert False, "Unknown model %s" % args.model

  model.init(X.shape[2], y.shape[2])
  cvfit(model, X, y, args)
