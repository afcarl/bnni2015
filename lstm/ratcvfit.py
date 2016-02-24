import numpy as np
import argparse
import math
import os.path
from ratlstm import *
from ratbilstm import *

def cvfit(model, X, y, k, save_path):
  n = X.shape[0]
  foldsize = math.ceil(n / float(k))

  # remember initial weights
  weights = model.get_weights()
  results = []
  for i in xrange(k):
    train_X = np.vstack((X[:i*foldsize], X[(i+1)*foldsize:]))
    train_y = np.vstack((y[:i*foldsize], y[(i+1)*foldsize:]))
    valid_X = X[i*foldsize:(i+1)*foldsize]
    valid_y = y[i*foldsize:(i+1)*foldsize]
    print "train: [:%d], [%d:], valid: [%d:%d]" % (i*foldsize, (i+1)*foldsize, i*foldsize, (i+1)*foldsize)
    print train_X.shape, train_y.shape, valid_X.shape, valid_y.shape

    model_path = save_path + "-" + str(i + 1) + ".hdf5"
    model.set_weights(weights)
    model.fit(train_X, train_y, valid_X, valid_y, model_path)
    result = model.eval(train_X, train_y, valid_X, valid_y, model_path)
    results.append(result)

  mean_dist = tuple(np.mean(results, axis=0))
  print "train mean dist = %g, valid mean dist = %g" % mean_dist
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
  cvfit(model, X, y, args.cvfolds, args.save_path)
