import numpy as np
import pickle

import tensorflow as tf
import numpy as np

import os
import sys
import yaml

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Set True if training on cluster
CLUSTER = config['CLUSTER']
GPUID = config['GPUID']
MEMORY = config['MEMORY']

if MEMORY > 12:
    sys.exit("Max memory is 12gb")

if CLUSTER:
  # SETUP FOR GPU
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=GPUID #select ID of GPU that shall be used

  # Choose memory
  # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
  gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=(MEMORY/12))

  sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#print("Using GPU", tf.test.gpu_device_name())

# CONFIGS
FEATURES = config['FEATURES']

BATCH_SIZE = config['BATCH_SIZE'] # Batch size
VERBOSE = config['VERBOSE'] # Output display mode
LENGTH = config['LENGTH'] # Packet sequence length

for features in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
  features = 'NoDef_{}'.format(features)
  #PATHS
  PATH_TO_DATA = config['PATH_TO_DATA']
  PATH_TO_MODEL = config['PATH_TO_MODEL']
  PATH_TO_EMBEDDINGS = config['PATH_TO_EMBEDDINGS']

  INPUT_SHAPE = (LENGTH,1)

  file_path = PATH_TO_DATA
  train_features = "/{}/X_train.pkl".format(features)
  train_labels = "/{}/y_train.pkl".format(features)
  valid_features = "/{}/X_valid.pkl".format(features)
  valid_labels = "/{}/y_valid.pkl".format(features)
  test_features = "/{}/X_test.pkl".format(features)
  test_labels = "/{}/y_test.pkl".format(features)

  # Load validation data
  with open(file_path + valid_features, 'rb') as handle:
    X_valid = np.array(pickle.load(handle, encoding='latin1'))
  with open(file_path + valid_labels, 'rb') as handle:
    y_valid = np.array(pickle.load(handle, encoding='latin1'))
  # Load test data
  with open(file_path + test_features, 'rb') as handle:
    X_test = np.array(pickle.load(handle, encoding='latin1'))
  with open(file_path + test_labels, 'rb') as handle:
    y_test = np.array(pickle.load(handle, encoding='latin1'))

  # Convert data as float32 type
  X_valid = X_valid.astype('float32')
  X_test = X_test.astype('float32')
  y_valid = y_valid.astype('float32')
  y_test = y_test.astype('float32')

  ###########################
  LENGTH = len(X_test[0])
  INPUT_SHAPE = (LENGTH,1)
  ###########################
  
  save_filename = PATH_TO_EMBEDDINGS + '/{0}-{1}.npz'.format('mu_{}'.format(features), 'raw')
  #print('Saving to', save_filename)

  np.savez(save_filename, X_valid=X_valid, X_test=X_test, y_valid=y_valid, y_test=y_test)

  for model_features_load in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]:
    model_features_load = 'NoDef_{}'.format(features)
    # Load model
    model_path = PATH_TO_MODEL + '/{}'.format(model_features_load)
    model = tf.keras.models.load_model(model_path)
    #model.summary()

    valid = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=VERBOSE)
    test = model.predict(X_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    save_filename = PATH_TO_EMBEDDINGS + '/{0}-{1}.npz'.format(features, model_features_load)
    #print('Saving to', save_filename)
    np.savez(save_filename, X_valid=valid, X_test=test, y_valid=y_valid, y_test=y_test)
