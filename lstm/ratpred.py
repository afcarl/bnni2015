import numpy as np
import argparse
import math
import os.path
from ratlstm import *
from ratbilstm import *

def cvpredict(model, X, k, save_path):
  n = X.shape[0]
  foldsize = math.ceil(n / float(k))
  (path, ext) = os.path.splitext(save_path)
  
  preds = []
  for i in xrange(k):
    valid_X = X[i*foldsize:(i+1)*foldsize]
    valid_y = y[i*foldsize:(i+1)*foldsize]
    print "valid: [%d:%d]" % (i*foldsize, (i+1)*foldsize)
    print valid_X.shape, valid_y.shape

    model_path = path + "-" + str(i + 1) + ext
    pred_y = model.predict(valid_X, model_path)
    preds.append(pred_y)

  return np.concatenate(preds)

if __name__ == '__main__':
  parser = create_parser()
  parser.add_argument("--cvfolds", type=int, default=5)
  parser.add_argument("--model", choices=['lstm', 'bilstm'], default='lstm')
  parser.add_argument("preds_path")
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
  preds = cvpredict(model, X, args.cvfolds, args.save_path)
  print "Preds: ", preds.shape
  np.save(args.preds_path, preds)
