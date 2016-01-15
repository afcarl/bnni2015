from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from ratdata import *
import numpy as np
import argparse

def mse(y, t, axis=2):
    return (np.square(y - t).mean(axis=axis).mean())

def mean_distance(y, t, axis=2):
    return np.mean(np.sqrt(np.sum((y - t)**2, axis=axis)))

class RatLSTM:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def init(self, nb_inputs, nb_outputs):
    print "Creating model..."
    self.model = Sequential()
    
    for i in xrange(self.layers):
      layer = LSTM(self.hidden_nodes, return_sequences=True, stateful=self.stateful, go_backwards=self.backwards)
      if i == 0:
        layer.set_input_shape((self.batch_size, self.seqlen, nb_inputs))
      self.model.add(layer)
      if self.dropout > 0:
        self.model.add(Dropout(self.dropout))
    self.model.add(TimeDistributedDense(nb_outputs))
    self.model.summary()
    
    print "Compiling model..."
    self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

  def fit(self, train_X, train_y, valid_X, valid_y, save_path):
    callbacks=[ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True), 
                   EarlyStopping(patience=self.patience, verbose=1)]
    if self.stateful:
      class StateReset(Callback):
          def on_epoch_begin(self, epoch, logs={}):
              print "Resetting states"
              self.model.reset_states()
      callbacks.append(StateReset())

    self.model.fit(train_X, train_y, batch_size=self.batch_size, nb_epoch=self.epochs, validation_data=(valid_X, valid_y), 
        shuffle=self.shuffle if self.shuffle=='batch' else True if self.shuffle=='true' else False, 
        verbose=self.verbose, callbacks=callbacks)

  def eval(self, train_X, train_y, valid_X, valid_y, load_path):
    self.model.load_weights(load_path)

    if self.stateful:
      self.model.reset_states()
    train_pred_y = self.model.predict(train_X, batch_size=1)

    if self.stateful:
      self.model.reset_states()
    valid_pred_y = self.model.predict(valid_X, batch_size=1)

    terr = mse(train_pred_y, train_y)
    verr = mse(valid_pred_y, valid_y)

    print 'train mse = %g, validation mse = %g' % (terr, verr)

    terr = mean_distance(train_pred_y, train_y)
    verr = mean_distance(valid_pred_y, valid_y)

    print 'train dist = %g, validation dist = %g' % (terr, verr)
    return (terr, verr)

  def get_weights(self):
    return self.model.get_weights()

  def set_weights(self, weights):
    return self.model.set_weights(weights)

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
  parser.add_argument("--backwards", action="store_true", default=False)
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

  model = RatLSTM(**vars(args))
  model.init(X.shape[2], y.shape[2])
  model.fit(train_X, train_y, valid_X, valid_y, args.save_path)
  model.eval(train_X, train_y, valid_X, valid_y, args.save_path)
