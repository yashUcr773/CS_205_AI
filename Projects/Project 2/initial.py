#####################################################################
###################            IMPORTS            ###################
#####################################################################

import numpy as np
import pandas as pd
from urllib import request
import zipfile
import time
import os

#####################################################################
################             GET DATASETS             ###############
#####################################################################

path_to_get_dataset_zip = 'https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_2_synthetic_dataset/data_sets.zip'

base_path = os.path.dirname(os.path.abspath(__file__))+'/datasets'

if not os.path.exists(base_path):
    os.makedirs(base_path)

path_to_store_dataset_zip = f'{base_path}/data_set.zip'
request.urlretrieve(path_to_get_dataset_zip, path_to_store_dataset_zip)

#####################################################################
################            UNZIP  DATASETS           ###############
#####################################################################

# https://docs.python.org/3/library/zipfile.html


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


unzip_file(path_to_store_dataset_zip, base_path)


#####################################################################
################         PATH TO TEST DATASETS        ###############
#####################################################################

# day of month of smallest {07/17/XXXX} = 17
small_dataset_path = 'CS170_small_Data__17.txt'

# day of month of largest {07/17/XXXX} = 17
large_dataset_path = 'CS170_large_Data__17.txt'

# sum of months of both {07/17/XXXX} = 14
xxx_large_dataset_path = 'CS170_XXXlarge_Data__14.txt'


#####################################################################
################           HELPER FUNCTIONS           ###############
#####################################################################

def train_test_partition(X, Y, ratio=0.2):

    # Combine X and Y into a single array for shuffling and splitting
    data = np.column_stack((X, Y))

    # Calculate the number of samples for testing
    num_test_samples = int(len(data) * ratio)

    # Randomly shuffle the data
    np.random.shuffle(data)

    # Split the data into training and testing sets
    x_train_set, y_train_set = data[:-num_test_samples,
                                    :-1], data[:-num_test_samples, -1]
    x_test_set, y_test_set = data[-num_test_samples:,
                                  :-1], data[-num_test_samples:, -1]

    return x_train_set, y_train_set, x_test_set, y_test_set


def euclidean(x1, x2):
    return np.sqrt(np.sum((x2 - x1)**2))

#####################################################################
################      NEAREST NEIGHBOR FUNCTION       ###############
#####################################################################


def knn(x_train, y_train, x_test, y_test):

    correct = 0
    for i in range(0, x_test.shape[0]):
        distances = []

        for j in range(0, x_train.shape[0]):
            distances.append((y_train[j], euclidean(x_train[j], x_test[i])))

        distances = np.array(sorted(distances, key=lambda x: x[1]))
        y_pred = distances[0][0]

        if y_pred == y_test[i]:
            correct += 1

    accuracy = correct/x_test.shape[0]
    return accuracy


#####################################################################
################           Cross Validation           ###############
#####################################################################

def k_fold_cross_validation(X, Y, k):

    fold_size = int(len(X) / k)
    accuracy_scores = []

    for i in range(k):
        fold_start = i * fold_size
        fold_end = fold_start + fold_size

        # Create the training set by excluding the current fold
        X_train = np.concatenate((X[:fold_start], X[fold_end:]), axis=0)
        Y_train = np.concatenate((Y[:fold_start], Y[fold_end:]), axis=0)

        # Create the validation set from the current fold
        X_test = X[fold_start:fold_end]
        Y_test = Y[fold_start:fold_end]

        # Get Test Accuracy
        accuracy = knn(X_train, Y_train, X_test, Y_test)

        # Append to list
        accuracy_scores.append(accuracy)

    # return average accuracy
    return np.mean(accuracy_scores)

#####################################################################
################          TEST ON TEST DATA           ###############
#####################################################################

################          DATA PREPROCESSING          ###############


test_datasets = ['CS170_small_Data__32.txt',
                 'CS170_small_Data__33.txt',
                 'CS170_large_Data__32.txt',
                 'CS170_large_Data__33.txt']

test_selected_features = [[3, 1, 5], [8, 7, 3], [3, 7, 6], [4, 5, 10]]
selection_index = 0

k = 2

while selection_index < len(test_datasets):
    df = pd.read_csv(f'{base_path}/{test_datasets[selection_index]}', sep='  ',
                     header=None, engine='python')
    print(
        f'The dataset {test_datasets[selection_index]} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])
    X = np.array(df[test_selected_features[selection_index]])

    t0 = time.time()
    k_fold_acc = k_fold_cross_validation(X, Y, k)
    t1 = time.time()

    print(
        f'k fold cross validation accuracy on {test_datasets[selection_index]} for k = {k} is {k_fold_acc:.3f}')
    print(f'It took {(t1 - t0):.3f} secs to run.\n')

    selection_index += 1


#####################################################################
################        TEST ON SELECTED DATA         ###############
#####################################################################

################                SMALL                 ###############

df = pd.read_csv(f'{base_path}/{small_dataset_path}', sep='  ',
                     header=None, engine='python')
print(
    f'The dataset {small_dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

k = df.shape[0]
X = np.array(df[list(range(1, df.shape[1]))])
Y = np.array(df[0])

t0 = time.time()
k_fold_acc = k_fold_cross_validation(X, Y, k)
t1 = time.time()

print(
    f'k fold cross validation accuracy on {small_dataset_path} for k = {k} is {k_fold_acc:.3f}')
print(f'It took {(t1 - t0):.3f} secs to run.\n')


################                LARGE                 ###############

df = pd.read_csv(f'{base_path}/{large_dataset_path}', sep='  ',
                     header=None, engine='python')
print(
    f'The dataset {large_dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

k = df.shape[0]
X = np.array(df[list(range(1, df.shape[1]))])
Y = np.array(df[0])

t0 = time.time()
k_fold_acc = k_fold_cross_validation(X, Y, k)
t1 = time.time()

print(
    f'k fold cross validation accuracy on {large_dataset_path} for k = {k} is {k_fold_acc:.3f}')
print(f'It took {(t1 - t0):.3f} secs to run.\n')


################                XXXLARGE                 ###############

df = pd.read_csv(f'{base_path}/{xxx_large_dataset_path}', sep='  ',
                     header=None, engine='python')
print(
    f'The dataset {xxx_large_dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

k = df.shape[0]
X = np.array(df[list(range(1, df.shape[1]))])
Y = np.array(df[0])

t0 = time.time()
k_fold_acc = k_fold_cross_validation(X, Y, k)
t1 = time.time()

print(
    f'k fold cross validation accuracy on {xxx_large_dataset_path} for k = {k} is {k_fold_acc:.3f}')
print(f'It took {(t1 - t0):.3f} secs to run.\n')