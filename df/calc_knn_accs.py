from absl import app
from absl import flags
from knn import *
import numpy as np
import os
import tensorflow as tf
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

#PATHS
# Cluster
PATH_TO_EMBEDDINGS = config['PATH_TO_EMBEDDINGS']



# Local
#PATH_TO_EMBEDDINGS = "/Users/alexanderveicht/Desktop/Bachelor_Thesis/Datasets.nosync/WCN+/LearnedFeatures/DF"

FLAGS = flags.FLAGS

FEATURES = config['FEATURES']


def main(argv):

    for features in FEATURES:
        minimum = 1
        
        for model_features_load in (['raw'] + FEATURES):
            filename = PATH_TO_EMBEDDINGS + '/{0}-{1}.npz'.format(features, model_features_load)
            files = np.load(filename)
            x_train = files['X_train']
            x_valid = files['X_valid']
            x_test = files['X_test']
            y_train = files['y_train']
            y_valid = files['y_valid']
            y_test = files['y_test']

            print(features, '--', model_features_load)
            print(" Shapes:", "Train Features:", x_train.shape, 'Valid Features', x_valid.shape, "Test Features:", x_test.shape, "Train Labels", y_train.shape, 'Valid Labels', y_valid.shape, "Test Labels", y_test.shape)
            results = eval_from_matrices(x_train, x_valid, y_train, y_valid)
            minimum = min(results['measure=squared_l2, k=1'][1], minimum)
            print(" Train/Valid-Results:", results)
            results = eval_from_matrices(x_valid, x_test, y_valid, y_test)
            minimum = min(results['measure=squared_l2, k=1'][1], minimum)
            print(" Valid/Test-Results:", results)
            results = eval_from_matrices(x_train, x_test, y_train, y_test)
            minimum = min(results['measure=squared_l2, k=1'][1], minimum)
            print(" Train/Test-Results:", results)
            print("")

        print('Min for {}: {:.4f}'.format(features, minimum))
        print("")

        print("++++")
        print("")

if __name__ == '__main__':
    app.run(main)
