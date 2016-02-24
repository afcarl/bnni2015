from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
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
    assert not self.stateful or self.batch_size == 1, "Stateful doesn't work with batch size > 1"
    assert not self.stateful or self.shuffle == 'false', "Stateful doesn't work with shuffle = true or shuffle = batch"

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
    callbacks = [ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=str2bool(self.save_best_model_only))]
    if self.patience:
      callbacks.append(EarlyStopping(patience=self.patience, verbose=1))
    if args.lr_epochs:
      def lr_scheduler(epoch):
        lr = args.lr / 2**int(epoch / args.lr_epochs)
        print "Epoch %d: learning rate %g" % (epoch + 1, lr)
        return lr
      callbacks.append(LearningRateScheduler(lr_scheduler))
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

  def predict(self, X, load_path):
    self.model.load_weights(load_path)

    if self.stateful:
      self.model.reset_states()
    pred_y = self.model.predict(X, batch_size=1)
    print 'pred_y.shape = ', pred_y.shape
    return pred_y

  def get_weights(self):
    return self.model.get_weights()

  def set_weights(self, weights):
    return self.model.set_weights(weights)

def add_model_params(parser):
  parser.add_argument("--hidden_nodes", type=int, default=1024)
  parser.add_argument("--batch_size", type=int, default=10)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--stateful", action="store_true", default=False)
  parser.add_argument("--backwards", action="store_true", default=False)
  parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
  parser.add_argument("--shuffle", choices=['batch', 'true', 'false'], default='true')
  parser.add_argument("--dropout", type=float, default=0.5)
  parser.add_argument("--layers", type=int, choices=[1, 2, 3], default=2)
  parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='rmsprop')
  parser.add_argument("--save_best_model_only", default="1")
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--lr_epochs", type=int)

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_data_params(parser)
  add_model_params(parser)
  parser.add_argument("save_path")
  args = parser.parse_args()

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args.seqlen)
  train_X, train_y, valid_X, valid_y = split_data(X, y, args.train_set)

  model = RatLSTM(**vars(args))
  model.init(X.shape[2], y.shape[2])
  model.fit(train_X, train_y, valid_X, valid_y, args.save_path)
  model.eval(train_X, train_y, valid_X, valid_y, args.save_path)
