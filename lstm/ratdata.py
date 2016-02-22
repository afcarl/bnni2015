from scipy.io import loadmat
import numpy as np

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
    y = np.loadtxt(locations)
  elif locations.endswith(".npy"):
    y = np.load(locations)
  else:
    assert False, "Unknown location file format"

  print "Original data: ", X.shape, y.shape, np.min(X), np.max(X), np.mean(X), np.std(X), np.min(y), np.max(y)
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
