import numpy as np
import argparse
from ratlstm import *
from ratbilstm import *

def cvfit(model, X, y, k, save_path):
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
    model.fit(train_X, train_y, valid_X, valid_y, save_path)
    result = model.eval(train_X, train_y, valid_X, valid_y, save_path)
    results.append(result)

  mean_dist = tuple(np.mean(results, axis=0))
  print "train mean dist = %g, valid mean dist = %g" % mean_dist
  return mean_dist

if __name__ == '__main__':
  parser = create_parser()
  parser.add_argument("--cvfolds", type=int, default=5)
  parser.add_argument("--model", choices=['lstm', 'bilstm'], default='bilstm')
  args = parser.parse_args()
  assert not args.stateful or args.batch_size == 1, "Stateful doesn't work with batch size > 1"

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
