import numpy as np
import argparse
from ratlstm import *
from ratbilstm import *

if __name__ == '__main__':
  parser = create_parser()
  parser.add_argument("--model", choices=['lstm', 'bilstm'], default='lstm')
  args = parser.parse_args()
  assert not args.stateful or args.batch_size == 1, "Stateful doesn't work with batch size > 1"

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args.seqlen)
  train_X, train_y, valid_X, valid_y = split_data(X, y, args.train_set)

  if args.model == 'lstm':
    model = RatLSTM(**vars(args))
  elif args.model == 'bilstm':
    model = RatBiLSTM(**vars(args))
  else:
    assert False, "Unknown model %s" % args.model

  model.init(X.shape[2], y.shape[2])
  model.eval(train_X, train_y, valid_X, valid_y, args.save_path)
