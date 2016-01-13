import numpy as np
import argparse
from ratgps_bilstm import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("save_path")
  parser.add_argument("--features", default="London_data_2x1000Center_bin100.dat")
  parser.add_argument("--locations", default="London_data_2x1000Center_bin100_pos.dat")
  parser.add_argument("--train_set", type=float, default=0.8)
  parser.add_argument("--seqlen", type=int, default=100)
  parser.add_argument("--hidden_nodes", type=int, default=512)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--stateful", action="store_true", default=False)
  parser.add_argument("--verbose", choices=[0, 1, 2], default=1)
  parser.add_argument("--folds", type=int, default=2)
  args = parser.parse_args()

  assert not args.stateful or args.batch_size == 1, "Stateful doesn't work with batch size > 1"

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args)
  train_X, train_y, valid_X, valid_y = split_data(X, y, args)
  model = create_model(X.shape[2], y.shape[2], args)
  #model = fit_data(model, train_X, train_y, valid_X, valid_y, args.save_path, args)
  print eval_data(model, train_X, train_y, valid_X, valid_y, args.save_path, args)
