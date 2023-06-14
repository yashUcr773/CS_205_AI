#####################################################################
###################            IMPORTS            ###################
#####################################################################

import numpy as np
import pandas as pd
from urllib import request
import zipfile
import time
import os
import matplotlib.pyplot as plt
import threading

#####################################################################
################             GET DATASETS             ###############
#####################################################################

path_to_get_dataset_zip = 'https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_2_synthetic_dataset/data_sets.zip'

try:
    print (__file__)
except:
    __file__ = './content'

print (__file__)
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


def print_formatted_time(time_input):
    '''
    funtion to take in seconds as input and
    print in Hours, minutes, and seconds
    '''
    hrs = int(time_input // 3600)
    mins = int((time_input % 3600) // 60)
    secs = int((time_input % 3600) % 60)
    if hrs:
        print(f'It took {hrs} hrs, {mins} mins and {secs} secs to run')
    elif mins:
        print(f'It took {mins} mins and {secs} secs to run')
    else:
        print(f'It took {secs} secs to run')


def print_time(time_input):
    '''
    function to print the time with appropritate precision if between 0 and 1
    else print in HH, MM, SS format
    '''
    if time_input <= 1e-5:
        print(f'It took {time_input:.6f} secs to run')
    elif time_input <= 1e-4:
        print(f'It took {time_input:.5f} secs to run')
    elif time_input <= 1e-3:
        print(f'It took {time_input:.4f} secs to run')
    elif time_input <= 1e-2:
        print(f'It took {time_input:.3f} secs to run')
    elif time_input <= 1e-1:
        print(f'It took {time_input:.2f} secs to run')
    elif time_input >= 0 and time_input <= 1:
        print(f'It took {time_input} secs to run')
    else:
        print_formatted_time(time_input)

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

def k_fold_cross_validation(X, Y, k, best_so_far, tolerence=5, verbose = False):

    accuracy_scores = []
    fold_size = len(X) // k

    # shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]

    if verbose:
        print (f'Total Instances: {X.shape[0]}, Total Features: {X.shape[1]}')
        print (f'K: {k}, Fold Size: {fold_size}')

    running_average = 0
    counter = 0

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
        running_average = np.mean(accuracy_scores)

        if len(accuracy_scores) > 5 and best_so_far != -1 and running_average < best_so_far:
            counter += 1
            if counter >= tolerence:
                return running_average

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

k_fold_k_value = 2

while selection_index < len(test_datasets):
    df = pd.read_csv(f'{base_path}/{test_datasets[selection_index]}', sep='  ',
                     header=None, engine='python')
    print(
        f'The dataset {test_datasets[selection_index]} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])
    X = np.array(df[test_selected_features[selection_index]])

    t0 = time.time()
    k_fold_acc = k_fold_cross_validation(X, Y, k_fold_k_value, best_so_far=-1, tolerence=0, verbose = False)
    t1 = time.time()

    print(
        f'k fold cross validation accuracy on {test_datasets[selection_index]} for k = {k_fold_k_value} is {k_fold_acc:.3f}')
    print(f'It took {(t1 - t0):.3f} secs to run.\n')

    selection_index += 1


#####################################################################
################        TEST ON SELECTED DATA         ###############
#####################################################################

for dataset_path in [small_dataset_path, large_dataset_path, xxx_large_dataset_path]:

    df = pd.read_csv(f'{base_path}/{dataset_path}', sep='  ',
                        header=None, engine='python')
    print(
        f'The dataset {dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    k = df.shape[0]
    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])

    # # Sampling if required
    # indices = np.random.permutation(len(X))
    # X = X[indices]
    # Y = Y[indices]
    # X = X[:1000]
    # Y = Y[:1000]
    # k = 2

    t0 = time.time()
    k_fold_acc = k_fold_cross_validation(X, Y, k, best_so_far=-1, tolerence=0, verbose = False)
    t1 = time.time()

    print(
        f'k fold cross validation accuracy on {dataset_path} for k = {k} is {k_fold_acc:.3f}')
    print_time(t1 - t0)
    print (f'The model will run for {(df.shape[1]*(df.shape[1]+1))/2} times and will take a total time of {((df.shape[1]*(df.shape[1]+1))/2)*(t1-t0)}')
    print_time(((df.shape[1]*(df.shape[1]+1))/2)*(t1-t0))

#####################################################################
################           FEATURE SELECTION          ###############
#####################################################################
for dataset_path in [small_dataset_path, large_dataset_path, xxx_large_dataset_path]:
# for dataset_path in ['CS170_small_Data__32.txt',
#                  'CS170_small_Data__33.txt',
#                  'CS170_large_Data__32.txt',
#                  'CS170_large_Data__33.txt']:

    df = pd.read_csv(f'{base_path}/{dataset_path}', sep='  ',
                        header=None, engine='python')
    print(
        f'The dataset {dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])
    # k_fold_k_value = 2
    k_fold_k_value = df.shape[0]

    values, counts = np.unique(Y, return_counts=True)
    default_rate = max(counts)/sum(counts)
    print (f'Default Rate is: {default_rate*100:.2f}%')

    selected_features = []
    accuracy_map = []
    accuracy_map.append((0, default_rate, []))
    best_so_far = default_rate
    best_features = []

    # run a loop from 0 to all features
    t00 = time.time()
    while len(selected_features) < df.shape[1] - 1:

        temp_acc_list = []
        time_per_feature = []
        verbose = True

        best_feature_accuracy = 0

        for i in range(1, df.shape[1]):
            if i in selected_features:
                continue

            new_features = list(selected_features)
            new_features.append(i)

            X = np.array(df[list(new_features)])
            Y = np.array(df[0])

            t0 = time.time()
            k_fold_acc = k_fold_cross_validation(X, Y, k_fold_k_value, best_so_far=-1, tolerence=0, verbose = verbose)


            t1 = time.time()
            if verbose == True:
                print (f'Time for 1 feature {t1 - t0}')

            verbose = False

            if k_fold_acc > best_feature_accuracy:
                best_feature_accuracy = k_fold_acc

            temp_acc_list.append((i, k_fold_acc))
            time_per_feature.append(t1-t0)



        temp_acc_list = sorted(temp_acc_list, reverse=True, key=lambda x:x[1])

        selected_features.append(temp_acc_list[0][0])
        accuracy_map.append((len(selected_features), temp_acc_list[0][1], list(selected_features)))
        print (f'Max accuracy of {temp_acc_list[0][1]*100:.2f}% for feature {temp_acc_list[0][0]} and the feature list is {selected_features}')
        print (f'Average time for each feature: {(sum(time_per_feature)/len(time_per_feature)):.2f}s')
        print (f'Total time for each feature: {(sum(time_per_feature)):.2f}s')

        if(temp_acc_list[0][1] > best_so_far):
            best_so_far = temp_acc_list[0][1]
            best_features = list(selected_features)

        print ()

    t11 = time.time()
    print_time(t11-t00)
    print ('----------------------------------------')

    print (f'Best Accuracy is {best_so_far:.2f}% with features {best_features}')

    accuracy_map = np.array(accuracy_map,  dtype=object)
    plt.figure(1,figsize=(25,5))
    plt.plot(accuracy_map[:,0], accuracy_map[:,1])
    plt.title('accuracy vs number of features')
    plt.xlabel('selected features')
    plt.ylabel('accuracy')
    plt.xticks(list(accuracy_map[:,0]), list([','.join([str(j) for j in i]) for i in accuracy_map[:,2]]))
    plt.grid()
    plt.plot()
    plt.show()

