import numpy as np
import argparse
from ratgps_bilstm import *

def cvfit(model, X, y, k, save_path, args):
  n = X.shape[0]
  foldsize = int(n / float(k))
  
  # remember initial weights
  weights = model.get_weights()
  results = []
  for i in xrange(k):
    train_X = np.vstack((X[:i*foldsize], X[(i+1)*foldsize:]))
    train_y = np.vstack((y[:i*foldsize], y[(i+1)*foldsize:]))
    valid_X = X[i*foldsize:(i+1)*foldsize]
    valid_y = y[i*foldsize:(i+1)*foldsize]
    #print "train: [:%d], [%d:], valid: [%d:%d]" % (i, i+foldsize, i, i+foldsize)
    #print train_X.shape, train_y.shape, valid_X.shape, valid_y.shape
    model.set_weights(weights)
    model = fit_data(model, train_X, train_y, valid_X, valid_y, save_path, args)
    result = eval_data(model, train_X, train_y, valid_X, valid_y, save_path, args)
    results.append(result)

  return tuple(np.mean(results, axis=0))

if __name__ == '__main__':
  parser = create_parser()
  parser.add_argument("--cvfolds", type=int, default=5)
  args = parser.parse_args()

  assert not args.stateful or args.batch_size == 1, "Stateful doesn't work with batch size > 1"

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args)
  model = create_model(X.shape[2], y.shape[2], args)
  train_mean_dist, valid_mean_dist = cvfit(model, X, y, args.cvfolds, args.save_path, args)
  print "train mean dist = %g, valid mean dist = %g" % (train_mean_dist, valid_mean_dist)
