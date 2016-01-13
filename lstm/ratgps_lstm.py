from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from scipy.io import loadmat
import numpy as np
import random
import sys
import argparse

def mse(y, t, axis=2):
    return (np.square(y - t).mean(axis=axis).mean())

def mean_distance(y, t, axis=2):
    return np.mean(np.sqrt(np.sum((y - t)**2, axis=axis)))

def load_data(features, locations):
  if features.endswith(".mat"):
    X = loadmat(features)
    X = X['mm'].T
  else:
    X = np.loadtxt(features)

  if locations.endswith(".mat"):
    y = loadmat(locations)
    y = y['loc'] / 3.5
  else:
    y = np.loadtxt(locations) / 3.5

  print "Original data: ", X.shape, y.shape, np.min(X), np.max(X), np.min(y), np.max(y)
  assert X.shape[0] == y.shape[0], "Number of samples in features and locations does not match"
  return (X, y)

def reshape_data(X, y, args):
  assert X.shape[0] == y.shape[0]
  nsamples = X.shape[0]
  nsamples = int(nsamples / args.seqlen) * args.seqlen

  # truncate remaining samples, if not divisible by sequence length
  X = X[:nsamples]
  y = y[:nsamples]

  nb_inputs = X.shape[1]
  nb_outputs = y.shape[1]

  X = np.reshape(X, (-1, args.seqlen, nb_inputs))
  y = np.reshape(y, (-1, args.seqlen, nb_outputs))

  print "After reshaping: ", X.shape, y.shape
  return (X, y)

def split_data(X, y, args):
  assert X.shape[0] == y.shape[0]
  nsamples = X.shape[0]

  if 0 <= args.train_set <= 1:
    ntrain = int(nsamples * args.train_set)
    nvalid = nsamples - ntrain
  else:
    ntrain = int(args.train_set)
    nvalid = nsamples - ntrain

  train_X = X[:ntrain]
  train_y = y[:ntrain]
  valid_X = X[ntrain:ntrain+nvalid]
  valid_y = y[ntrain:ntrain+nvalid]

  print "After splitting: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape
  return (train_X, train_y, valid_X, valid_y)

def create_model(nb_inputs, nb_outputs, args):
  print "Creating model..."
  model = Sequential()
  model.add(LSTM(args.hidden_nodes, return_sequences=True, batch_input_shape=(args.batch_size, args.seqlen, nb_inputs), stateful=args.stateful))
  #model.add(Dropout(0.2))
  #model.add(LSTM(512, return_sequences=True))
  #model.add(Dropout(0.2))
  model.add(TimeDistributedDense(nb_outputs))
  
  print "Compiling model..."
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

def fit_data(model, train_X, train_y, valid_X, valid_y, save_path, args):
  callbacks=[ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True), 
                 EarlyStopping(patience=args.patience, verbose=1)]
  if args.stateful:
    class StateReset(Callback):
        def on_epoch_begin(self, epoch, logs={}):
            print "Resetting states"
            self.model.reset_states()
    callbacks.append(StateReset())

  model.fit(train_X, train_y, batch_size=args.batch_size, nb_epoch=args.epochs, validation_data=(valid_X, valid_y), 
      shuffle=False if args.stateful else 'batch', verbose=args.verbose, callbacks=callbacks)

  return model

def eval_data(model, train_X, train_y, valid_X, valid_y, load_path, args):
  model.load_weights(load_path)

  if args.stateful:
    model.reset_states()
  train_pred_y = model.predict(train_X, batch_size=1)

  if args.stateful:
    model.reset_states()
  valid_pred_y = model.predict(valid_X, batch_size=1)

  terr = mean_distance(train_pred_y, train_y)
  verr = mean_distance(valid_pred_y, valid_y)

  print 'train dist = %g, validation dist = %g' % (terr, verr)
  return (terr, verr)

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
  model = fit_data(model, train_X, train_y, valid_X, valid_y, args.save_path, args)
  print eval_data(model, train_X, train_y, valid_X, valid_y, args.save_path, args)
