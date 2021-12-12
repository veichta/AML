import numpy as np
import pickle

import tensorflow as tf
import os
import sys
import yaml

import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import Callback


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
  print('[INFO] using GPU with id: {} ({})'.format(GPUID, tf.test.is_gpu_available()))

#print("Using GPU", tf.test.gpu_device_name())

# CONFIGS
FEATURES = config['FEATURES']

for feature in FEATURES:


  #print('[*****] Training model for: ', feature)
  experiment_name = feature
  #sigma = feature.split('_')[1]
  #rho = feature.split('_')[2]

  wandb.init(
    project='DF-BER',
    entity='veichta',
    group="Dynaflow",
    config={
      "epoch": config['NB_EPOCH'],
      "batch_size": config['BATCH_SIZE'],
      "n_traces": 1000,
      "feature": feature,
      #"sigma": float(sigma),
      #"rho": float(rho)
    }
  )
  
  #PATHS
  PATH_TO_DATA = config['PATH_TO_DATA']
  PATH_TO_MODEL = config['PATH_TO_MODEL'] + '/{}/'.format(feature)
  
  NB_EPOCH = config['NB_EPOCH']   # Number of training epoch (suggested 40)
  BATCH_SIZE = config['BATCH_SIZE'] # Batch size
  VERBOSE = config['VERBOSE'] # Output display mode
  LENGTH = config['LENGTH'] # Packet sequence length
  OPTIMIZER = tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

  NB_CLASSES = config['NB_CLASSES'] # number of outputs = number of classes for WalkieTalkie and all WCN+
  INPUT_SHAPE = (LENGTH,1)

  train_features = "/{}/X_train.pkl".format(feature)
  train_labels = "/{}/y_train.pkl".format(feature)
  valid_features = "/{}/X_valid.pkl".format(feature)
  valid_labels = "/{}/y_valid.pkl".format(feature)
  test_features = "/{}/X_test.pkl".format(feature)
  test_labels = "/{}/y_test.pkl".format(feature)

  # Load training data
  with open(PATH_TO_DATA + train_features, 'rb') as handle:
    X_train = np.array(pickle.load(handle, encoding='latin1'))
  with open(PATH_TO_DATA + train_labels, 'rb') as handle:
    y_train = np.array(pickle.load(handle, encoding='latin1'))
  # Load validation data
  with open(PATH_TO_DATA + valid_features, 'rb') as handle:
    X_valid = np.array(pickle.load(handle, encoding='latin1'))
  with open(PATH_TO_DATA + valid_labels, 'rb') as handle:
    y_valid = np.array(pickle.load(handle, encoding='latin1'))
  # Load test data
  with open(PATH_TO_DATA + test_features, 'rb') as handle:
    X_test = np.array(pickle.load(handle, encoding='latin1'))
  with open(PATH_TO_DATA + test_labels, 'rb') as handle:
    y_test = np.array(pickle.load(handle, encoding='latin1'))


  # Convert data as float32 type
  X_train = X_train.astype('float32')
  X_valid = X_valid.astype('float32')
  X_test = X_test.astype('float32')
  y_train = y_train.astype('float32')
  y_valid = y_valid.astype('float32')
  y_test = y_test.astype('float32')

  print(X_train.shape, 'train samples shape')
  print(X_valid.shape, 'validation samples shape')
  print(X_test.shape, 'test samples shape')
  
  # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
  X_train = X_train[:, :,np.newaxis]
  X_valid = X_valid[:, :,np.newaxis]
  X_test = X_test[:, :,np.newaxis]

  
  print(X_train.shape, 'train samples shape')
  print(X_valid.shape, 'validation samples shape')
  print(X_test.shape, 'test samples shape')
  
  def build_model(input_shape, classes):

    pool_stride_size = ['None',4,4,4,4]
    pool_size = ['None',8,8,8,8]

    m = tf.keras.Sequential()

    for i, f in enumerate([32, 64, 128, 256]):
      m.add(tf.keras.layers.Conv1D(filters=f, kernel_size=8, input_shape=input_shape, strides=1, padding='same'))
      m.add(tf.keras.layers.BatchNormalization(axis=-1))
      if i == 0:
        m.add(tf.keras.layers.ELU(alpha=1.0))
      else:
        m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Conv1D(filters=f, kernel_size=8, input_shape=input_shape, strides=1, padding='same'))
      m.add(tf.keras.layers.BatchNormalization(axis=-1))
      if i == 0:
        m.add(tf.keras.layers.ELU(alpha=1.0))
      else:
        m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPool1D(8, 4, padding='same'))
      m.add(tf.keras.layers.MaxPool1D(8, 4, padding='same'))
      m.add(tf.keras.layers.Dropout(0.2))

    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    m.add(tf.keras.layers.Dropout(0.7))

    m.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    m.add(tf.keras.layers.Dropout(0.5))

    m.add(tf.keras.layers.Dense(classes, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)))
    m.add(tf.keras.layers.Activation('softmax'))

    m.build([None, input_shape])

    return m

  model = build_model(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
  #model.summary()  
  
  model.compile(loss="sparse_categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

  history = model.fit(X_train, y_train,
      batch_size=BATCH_SIZE, 
      epochs=NB_EPOCH,
      verbose=VERBOSE, 
      validation_data=(X_valid, y_valid),
      #callbacks=[WandbCallback()]
  )

  

  #print("Start evaluating model with testing data")
  score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)

  for epoch in range(NB_EPOCH):
    wandb.log({
      'epoch': epoch,
      'loss': history.history['loss'][epoch],
      'val_loss': history.history['loss'][epoch],
      'accuracy': history.history['accuracy'][epoch],
      'val_accuracy': history.history['val_accuracy'][epoch]
    })

  wandb.log({
    'test_accuracy': score_test[1]
  })

  print("[***] Testing accuracy for {}: {}".format(feature, score_test[1]))

  # Save part of model:
  model.pop() # ReLU
  model.pop() # Dropout
  model.pop() # Dense
  model.pop() # Softmax
  #model.summary()

  print("Saving model to: ", PATH_TO_MODEL)
  #tf.saved_model(model, PATH_TO_MODEL)
  model.save(PATH_TO_MODEL)
  
  wandb.finish()
