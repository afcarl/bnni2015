from keras.models import Graph
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
  elif features.endswith(".dat"):
    X = np.loadtxt(features)
  elif features.endswith(".npy"):
    X = np.load(features)
  else:
    assert False, "Unknown feature file format"

  if locations.endswith(".mat"):
    y = loadmat(locations)
    y = y['loc'] / 3.5
  elif locations.endswith(".dat"):
    y = np.loadtxt(locations) / 3.5
  elif locations.endswith(".npy"):
    y = np.load(locations)
  else:
    assert False, "Unknown location file format"

  print "Original data: ", X.shape, y.shape, np.min(X), np.max(X), np.min(y), np.max(y)
  assert X.shape[0] == y.shape[0], "Number of samples in features and locations does not match"
  return (X, y)

def reshape_data(X, y, seqlen):
  assert X.shape[0] == y.shape[0]
  nsamples = X.shape[0]
  nsamples = int(nsamples / seqlen) * seqlen

  # truncate remaining samples, if not divisible by sequence length
  X = X[:nsamples]
  y = y[:nsamples]

  nb_inputs = X.shape[1]
  nb_outputs = y.shape[1]

  X = np.reshape(X, (-1, seqlen, nb_inputs))
  y = np.reshape(y, (-1, seqlen, nb_outputs))

  print "After reshaping: ", X.shape, y.shape
  return (X, y)

def split_data(X, y, train_set):
  assert X.shape[0] == y.shape[0]
  nsamples = X.shape[0]

  if 0 <= train_set <= 1:
    ntrain = int(nsamples * train_set)
    nvalid = nsamples - ntrain
  else:
    ntrain = int(train_set)
    nvalid = nsamples - ntrain

  train_X = X[:ntrain]
  train_y = y[:ntrain]
  valid_X = X[ntrain:ntrain+nvalid]
  valid_y = y[ntrain:ntrain+nvalid]

  print "After splitting: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape
  return (train_X, train_y, valid_X, valid_y)

def create_model(nb_inputs, nb_outputs, args):
  print "Creating model..."
  model = Graph()
  model.add_input(name='input', batch_input_shape=(args.batch_size, args.seqlen, nb_inputs))
  for i in xrange(args.layers):
    model.add_node(LSTM(args.hidden_nodes, return_sequences=True, stateful=args.stateful), name='forward'+str(i+1), 
        input='input' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
        inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
    print "name=", 'forward'+str(i+1), "input=", 'input' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, "inputs=", ['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else []
    model.add_node(LSTM(args.hidden_nodes, return_sequences=True, stateful=args.stateful, go_backwards=True), name='backward'+str(i+1), 
        input='input' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
        inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
    print "name=", 'backward'+str(i+1), "input=", 'input' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, "inputs=", ['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else []
    if args.dropout > 0:
      model.add_node(Dropout(args.dropout), name='dropout'+str(i+1), inputs=['forward'+str(i+1), 'backward'+str(i+1)])
      print "name=", 'dropout'+str(i+1), "inputs=", ['forward'+str(i+1), 'backward'+str(i+1)]

  model.add_node(TimeDistributedDense(nb_outputs), name='dense', 
      input='dropout'+str(args.layers) if args.dropout > 0 else None,
      inputs=['forward'+str(args.layers), 'backward'+str(args.layers)] if args.dropout == 0 else [])
  print "name=", 'dense', "input=", 'dropout'+str(args.layers) if args.dropout > 0 else None, "inputs=", ['forward'+str(args.layers), 'backward'+str(args.layers)] if args.dropout == 0 else []
  model.add_output(name='output', input='dense')
  print "name=", 'output', "input=", 'dense'
  model.summary()

  print "Compiling model..."
  model.compile(args.optimizer, {'output': 'mean_squared_error'})
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

  model.fit({'input': train_X, 'output': train_y}, batch_size=args.batch_size, nb_epoch=args.epochs, 
      validation_data={'input': valid_X, 'output': valid_y}, 
      shuffle=args.shuffle if args.shuffle=='batch' else True if args.shuffle=='true' else False,
      verbose=args.verbose, callbacks=callbacks)

  return model

def eval_data(model, train_X, train_y, valid_X, valid_y, load_path, args):
  model.load_weights(load_path)

  if args.stateful:
    model.reset_states()
  train_pred_y = model.predict({'input': train_X}, batch_size=1)['output']

  if args.stateful:
    model.reset_states()
  valid_pred_y = model.predict({'input': valid_X}, batch_size=1)['output']

  terr = mean_distance(train_pred_y, train_y)
  verr = mean_distance(valid_pred_y, valid_y)

  print 'train dist = %g, validation dist = %g' % (terr, verr)
  return (terr, verr)

def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("save_path")
  parser.add_argument("--features", default="London_data_2x1000Center_bin100.dat")
  parser.add_argument("--locations", default="London_data_2x1000Center_bin100_pos.dat")
  parser.add_argument("--train_set", type=float, default=0.8)
  parser.add_argument("--seqlen", type=int, default=100)
  parser.add_argument("--hidden_nodes", type=int, default=1024)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--stateful", action="store_true", default=False)
  #parser.add_argument("--backwards", action="store_true", default=False)
  parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
  parser.add_argument("--shuffle", choices=['batch', 'true', 'false'], default='true')
  parser.add_argument("--dropout", type=float, default=0.5)
  parser.add_argument("--layers", type=int, choices=[1, 2, 3], default=1)
  parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='rmsprop')
  return parser

if __name__ == '__main__':
  parser = create_parser()
  args = parser.parse_args()

  assert not args.stateful or args.batch_size == 1, "Stateful doesn't work with batch size > 1"

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args.seqlen)
  train_X, train_y, valid_X, valid_y = split_data(X, y, args.train_set)
  model = create_model(X.shape[2], y.shape[2], args)
  model = fit_data(model, train_X, train_y, valid_X, valid_y, args.save_path, args)
  print eval_data(model, train_X, train_y, valid_X, valid_y, args.save_path, args)
