import numpy as np
import argparse
import math
import os.path
from ratlstm import *
from ratbilstm import *

def cvpredict(model, X, k, save_path):
  n = X.shape[0]
  foldsize = math.ceil(n / float(k))
  
  preds = []
  for i in xrange(k):
    valid_X = X[i*foldsize:(i+1)*foldsize]
    valid_y = y[i*foldsize:(i+1)*foldsize]
    print "valid: [%d:%d]" % (i*foldsize, (i+1)*foldsize)
    print valid_X.shape, valid_y.shape

    model_path = save_path + "-" + str(i + 1) + ".hdf5"
    pred_y = model.predict(valid_X, model_path)
    preds.append(pred_y)

  return np.concatenate(preds)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_data_params(parser)
  add_model_params(parser)
  parser.add_argument("save_path")
  parser.add_argument("preds_path")
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
  preds = cvpredict(model, X, args.cvfolds, args.save_path)
  print "Preds: ", preds.shape
  np.save(args.preds_path, preds)
