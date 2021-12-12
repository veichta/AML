import argparse

import numpy as np
import pickle

from absl import app
from absl import flags
from knn import *

import tensorflow as tf

import os
from tabulate import tabulate
from sklearn.metrics import classification_report
#from sklearn.model_selection import KFold

import wandb
import pandas as pd

FLAGS = flags.FLAGS

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # What to run
    parser.add_argument('--estimate', default='all', type=str, required=True)
    
    # Paths and data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--features', nargs='+', type=str, required=True)
    
    # Training
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--featrue_length', default=5000, type=int)
    #parser.add_argument('--k_fold', default=5, type=int) maybe include KFold for later
    
    # Cluster
    parser.add_argument('--gpu_id', default=0, type=str)
    parser.add_argument('--gpu_mem', default=11, type=float)

    # Logging
    parser.add_argument('--use_wandb', default=0, type=int)
    parser.add_argument('--log_group', default='', type=str)
    parser.add_argument('--log_dir', default='', type=str)
    parser.add_argument('--verbose', default=1, type=int)

    # Knn
    parser.add_argument('--knn_measure', default="squared_l2", choices=["squared_l2", "cosine"], type=str, help="Measure used for KNN distance matrix computation")
    parser.add_argument('--knn_k', default=1, nargs='+', type=int, help="Values for K in KNN")
    parser.add_argument('--knn_subtest', default=None, type=int, help="Split the testset for estimation")
    parser.add_argument('--knn_subtrain', default=None, type=int, help="Split the testset for estimation")
    
    return parser


def log_info(message):
    if args.log_dir != '':
        with open(args.log_dir + 'log.txt', 'a+') as f:
            f.write(f'[INFO] {message}' + "\n")
    else:
        print(f'[INFO] {message}')


def log_error(message):
    if args.log_dir != '':
        with open(args.log_dir + 'log.txt', 'a+') as f:
            f.write(f'[ERROR] {message}' + "\n")
    else:
        print(f'[ERROR] {message}')


def load_data(feature, args):
    train_features = "/{}/X_train.pkl".format(feature)
    train_labels = "/{}/y_train.pkl".format(feature)
    valid_features = "/{}/X_valid.pkl".format(feature)
    valid_labels = "/{}/y_valid.pkl".format(feature)
    test_features = "/{}/X_test.pkl".format(feature)
    test_labels = "/{}/y_test.pkl".format(feature)

    # Load training data
    with open(args.data_path + train_features, 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding='latin1'))
    with open(args.data_path + train_labels, 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='latin1'))
    # Load validation data
    with open(args.data_path + valid_features, 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='latin1'))
    with open(args.data_path + valid_labels, 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='latin1'))
    # Load test data
    with open(args.data_path + test_features, 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='latin1'))
    with open(args.data_path + test_labels, 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='latin1'))


    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')
    y_test = y_test.astype('float32')

    if args.verbose == 1:
        log_info(f'loaded {X_train.shape[0]}, train samples')
        log_info(f'loaded {X_valid.shape[0]}, valid samples')
        log_info(f'loaded {X_test.shape[0]}, test samples')

    # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
    X_train = X_train[:, :,np.newaxis]
    X_valid = X_valid[:, :,np.newaxis]
    X_test = X_test[:, :,np.newaxis]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


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


def train_df_model(args):
    log_info('------------------- Training -------------------')
    summary = {}
    for feature in list(args.features):
        if args.use_wandb:
            wandb.init(
                project='DF-BER',
                entity='veichta',
                group=args.log_group,
                config={
                    'feature': feature,
                    'epochs': args.epochs,
                }
            )
    
        # load data of shape (num_samples, trace_len, 1)
        log_info(f'loading {feature} data ...')
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(feature, args)

        # build model
        model = build_model(input_shape=(args.featrue_length,1), classes=args.num_classes)
        
        # optimizer
        optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # compile model
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # fit model
        log_info(f'start training model for feature {feature} ...')
        history = model.fit(X_train, y_train,
            batch_size=args.batch_size, 
            epochs=args.epochs,
            verbose=args.verbose, 
            validation_data=(X_valid, y_valid),
        )

        log_info('evaluating the model ...')
        score_train = model.evaluate(X_train, y_train, verbose=args.verbose)
        score_valid = model.evaluate(X_valid, y_valid, verbose=args.verbose)
        score_test = model.evaluate(X_test, y_test, verbose=args.verbose)
        
        y_hat_test = np.argmax(model.predict(X_test), axis=-1)
        log_info(f'------------------- Test Report -------------------\n{classification_report(y_test, y_hat_test)}')
        
        y_hat_valid = np.argmax(model.predict(X_valid), axis=-1)
        log_info(f'------------------- Valid Report -------------------\n{classification_report(y_valid, y_hat_valid)}')
        summary[feature] = [score_train[1], score_valid[1], score_test[1]]


        # logging
        if args.use_wandb:
            for epoch in range(args.epochs):
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
        
        
        log_info(f"Training accuracy for {feature} is {score_train[1]}")
        log_info(f"Validation accuracy for {feature} is {score_valid[1]}")
        log_info(f"Testing accuracy for {feature} is {score_test[1]}")

        # remove the last layers of the model and save it
        model.pop() # ReLU
        model.pop() # Dropout
        model.pop() # Dense
        model.pop() # Softmax

        # save model
        model_path = os.path.join(args.model_path, feature)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            log_info(f'create path and saving model to {model_path}')

        model.save(model_path)

        if args.use_wandb:
            wandb.finish()


    table = tabulate([[elem for elem in [k] + summary[k]] for k in summary.keys()], headers=['feature', 'train_acc', 'valid_acc', 'test_acc'])
    log_info(f'------------------- Summary for Accuracy -------------------\n{table}\n')

    if args.log_dir != '':
        df = pd.DataFrame(summary).T
        df.columns = ['train_acc', 'valid_acc', 'test_acc']
        df.to_csv(args.log_dir + 'train_summary.csv')



def extract_embeddings(args):
    log_info('------------------- EMBEDDINGS -------------------')
    log_info(f'saving to {args.embedding_path}')
    summary = []
    for feature in list(args.features):
        # load data of shape (num_samples, trace_len, 1) and ignore training data
        log_info(f'loading {feature} data ...')
        _, X_valid, X_test, _, y_valid, y_test = load_data(feature, args)

        # save raw traces
        save_filename = os.path.join(args.embedding_path, f'{feature}-raw.npz')
        if not os.path.exists(args.embedding_path):
                os.mkdir(args.embedding_path)
                log_info(f'create path {args.embedding_path}')

        summary.append(f'{feature}-raw.npz')
        np.savez(save_filename, X_valid=X_valid, X_test=X_test, y_valid=y_valid, y_test=y_test)

        # extract embeddings
        for model_features_load in list(args.features):

            model_path = os.path.join(args.model_path, f'{model_features_load}')
            model = tf.keras.models.load_model(model_path)

            valid = model.predict(X_valid, batch_size=args.batch_size, verbose=args.verbose)
            test = model.predict(X_test, batch_size=args.batch_size, verbose=args.verbose)

            save_filename = os.path.join(args.embedding_path,  f'{feature}-{model_features_load}.npz')
            summary.append(f'{feature}-{model_features_load}.npz')
            np.savez(save_filename, X_valid=valid, X_test=test, y_valid=y_valid, y_test=y_test)
        
    table = tabulate([[s] for s in summary], headers=['created embeddings'])
    log_info(f'------------------- Summary Embeddings -------------------:\n{table}\n')


def calc_bounds(args):
    log_info('------------------- ESTIMATING BOUNDS -------------------')
    overall_summary = {}
    for feature in list(args.features):
        summary = {}
        curr_min = 1
        for model_features_load in (['raw'] + list(args.features)):
            filename = os.path.join(args.embedding_path,  f'{feature}-{model_features_load}.npz')
            files = np.load(filename)
            
            x_valid = files['X_valid']
            x_test = files['X_test']
            y_valid = files['y_valid']
            y_test = files['y_test']

            x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

            # valid for train
            results = eval_from_matrices(x_valid, x_test, y_valid, y_test, args)
            knn_est_valid = results['measure=squared_l2, k=1'][1]
            curr_min = min(curr_min, knn_est_valid)
            

            # test for train
            results = eval_from_matrices(x_test, x_valid, y_test, y_valid, args)
            knn_est_test = results['measure=squared_l2, k=1'][1]
            curr_min = min(curr_min, knn_est_valid)

            summary[model_features_load] = [knn_est_test, knn_est_valid]
        
        overall_summary[f'{feature}'] = curr_min
        table = tabulate([[s for s in [k] + summary[k]] for k in summary.keys()], headers=['Model Feature Map', 'Test Bayes Error Rate', 'Valid Bayes Error Rate'])
        log_info(f'------------------- Summary {feature} Embeddings -------------------:\n{table}\n')
        df = pd.DataFrame(summary).T
        df.columns = ['test_ber', 'valid_ber']
        
        if args.log_dir != '':
            if not os.path.exists(args.log_dir + f'knn/'):
                os.mkdir(args.log_dir + f'knn/')
            df.to_csv(args.log_dir + f'knn/{feature}.csv')
    

    table = tabulate([[k, overall_summary[k], 1-overall_summary[k]] for k in overall_summary.keys()], headers=['Feature', 'Bayes Error Rate (min)', 'Maximum Accuracy'])
    log_info(f'------------------- Summary for BER Estimation -------------------:\n{table}\n')

    for k in overall_summary.keys():
        overall_summary[k] = [overall_summary[k], 1-overall_summary[k]]
    if args.log_dir != '':
        df = pd.DataFrame(overall_summary).T
        df.columns = ['ber_min', 'acc_max']
        df.to_csv(args.log_dir + 'ber_summary.csv')


def main(args):

    if args.estimate == 'all':
        train_df_model(args)
        extract_embeddings(args)
        calc_bounds(args)

    elif args.estimate == 'acc':
        train_df_model(args)

    elif args.estimate == 'ber':
        extract_embeddings(args)
        calc_bounds(args)

    else:
        log_error('estimate neets to be in [all, acc, ber]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if not os.path.exists(args.log_dir) and args.log_dir != '':
        os.mkdir(args.log_dir)


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id #select ID of GPU that shall be used

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=(args.gpu_mem/12))

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    print('[INFO] using GPU with id: {} ({})'.format(args.gpu_id, len(tf.config.list_physical_devices('GPU'))>0))
    
    main(args)
