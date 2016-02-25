from keras.models import Graph
from keras.layers.core import TimeDistributedDense, Activation, Dropout, Lambda
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from ratdata import *
from ratlstm import *
import numpy as np
import argparse

def reverse_func(x):
  import keras.backend as K
  assert K.ndim(x) == 3, "Should be a 3D tensor."
  rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
  return K.permute_dimensions(rev, (1, 0, 2))

class RatBiLSTM(RatLSTM):
  def __init__(self, **kwargs):
    RatLSTM.__init__(self, **kwargs)

  def init(self, nb_inputs, nb_outputs):
    print "Creating model..."
    self.model = Graph()
    self.model.add_input(name='input', batch_input_shape=(self.batch_size, self.seqlen, nb_inputs))
    for i in xrange(self.layers):
      self.model.add_node(LSTM(self.hidden_nodes, return_sequences=True, stateful=self.stateful), name='forward'+str(i+1), 
          input='input' if i == 0 else 'dropout'+str(i) if self.dropout > 0 else None, 
          inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and self.dropout == 0 else [])
      self.model.add_node(LSTM(self.hidden_nodes, return_sequences=True, stateful=self.stateful, go_backwards=True), name='backwarda'+str(i+1), 
          input='input' if i == 0 else 'dropout'+str(i) if self.dropout > 0 else None, 
          inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and self.dropout == 0 else [])
      self.model.add_node(Lambda(reverse_func), name='backward'+str(i+1), input='backwarda'+str(i+1)) # reverse backwards sequence
      if self.dropout > 0:
        self.model.add_node(Dropout(self.dropout), name='dropout'+str(i+1), inputs=['forward'+str(i+1), 'backward'+str(i+1)])

    self.model.add_node(TimeDistributedDense(nb_outputs), name='dense', 
        input='dropout'+str(self.layers) if self.dropout > 0 else None,
        inputs=['forward'+str(self.layers), 'backward'+str(self.layers)] if self.dropout == 0 else [])
    self.model.add_output(name='output', input='dense')
    self.model.summary()

    print "Compiling model..."
    self.model.compile(self.optimizer, {'output': 'mean_squared_error'})

  def fit(self, train_X, train_y, valid_X, valid_y, save_path):
    callbacks=[ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True), 
                   EarlyStopping(patience=self.patience, verbose=1)]
    if self.stateful:
      class StateReset(Callback):
          def on_epoch_begin(self, epoch, logs={}):
              print "Resetting states"
              self.model.reset_states()
      callbacks.append(StateReset())

    self.model.fit({'input': train_X, 'output': train_y}, batch_size=self.batch_size, nb_epoch=self.epochs, 
        validation_data={'input': valid_X, 'output': valid_y}, 
        shuffle=self.train_shuffle if self.train_shuffle=='batch' else True if self.train_shuffle=='true' else False,
        verbose=self.verbose, callbacks=callbacks)

  def eval(self, X, y, load_path):
    self.model.load_weights(load_path)

    if self.stateful:
      self.model.reset_states()
    pred_y = self.model.predict({'input': X}, batch_size=1)['output']

    err = mse(pred_y, y)
    dist = mean_distance(pred_y, y)

    return (err, dist)

  def predict(self, X, load_path):
    self.model.load_weights(load_path)

    if self.stateful:
      self.model.reset_states()
    pred_y = self.model.predict({'input': X}, batch_size=1)['output']
    print 'pred_y.shape = ', pred_y.shape
    return pred_y

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_data_params(parser)
  add_model_params(parser)
  parser.add_argument("save_path")
  args = parser.parse_args()

  X, y = load_data(args.features, args.locations)
  X, y = reshape_data(X, y, args.seqlen)
  train_X, train_y, valid_X, valid_y = split_data(X, y, args.train_set, args.split_shuffle)

  model = RatBiLSTM(**vars(args))
  model.init(X.shape[2], y.shape[2])
  model.fit(train_X, train_y, valid_X, valid_y, args.save_path + '.hdf5')
  terr, tdist = model.eval(train_X, train_y, args.save_path + '.hdf5')
  verr, vdist = model.eval(valid_X, valid_y, args.save_path + '.hdf5')
  print 'train mse = %g, validation mse = %g' % (terr, verr)
  print 'train dist = %g, validation dist = %g' % (tdist, vdist)
