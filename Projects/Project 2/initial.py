#####################################################################
###################            IMPORTS            ###################
#####################################################################

# To import and use dataset
import pandas as pd

# To work with pandas dataframe and plot graphs easily
import numpy as np

# To download the Dataset from cloud
from urllib import request

# To extract the dataset
import zipfile

# To time how long the execution takes place
import time

# To work with filesystem for extracting and locating the dataset
import os

# To plot the graphs
import matplotlib.pyplot as plt

# To get Real world dataset - Penguins Dataset
import seaborn as sns

# To convert categorical data into numerical data in real world dataset.
from sklearn.preprocessing import LabelEncoder

"""### Get and Setup Dataset"""

#####################################################################
################             GET DATASETS             ###############
#####################################################################
'''
Code the download the dataset.
The dataset is hosted on CDN so that it is available for everyone to download
'''

path_to_get_dataset_zip = 'https://d1u36hdvoy9y69.cloudfront.net/cs-205-ai/Project_2_synthetic_dataset/data_sets.zip'

# Hack to make it runnable in google colab as it does not support __file__
try:
    print(__file__)
except:
    __file__ = './content'

print(__file__)
base_path = os.path.dirname(os.path.abspath(__file__))+'/datasets'

# create the dataset directory if it does not exist
if not os.path.exists(base_path):
    os.makedirs(base_path)

# download the dataset and store it.
path_to_store_dataset_zip = f'{base_path}/data_set.zip'
request.urlretrieve(path_to_get_dataset_zip, path_to_store_dataset_zip)

#####################################################################
################            UNZIP  DATASETS           ###############
#####################################################################

# https://docs.python.org/3/library/zipfile.html

'''
Unzip the dataset into corresponding csv files
'''


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


unzip_file(path_to_store_dataset_zip, base_path)

#####################################################################
################         PATH TO TEST DATASETS        ###############
#####################################################################

'''
As I am solo and my DOB is 07/17/XXXX,
Select the datasets accordingly
'''

# day of month of smallest {07/17/XXXX} = 17
small_dataset_path = 'CS170_small_Data__17.txt'

# day of month of largest {07/17/XXXX} = 17
large_dataset_path = 'CS170_large_Data__17.txt'

# sum of months of both {07/17/XXXX} = 14
xxx_large_dataset_path = 'CS170_XXXlarge_Data__14.txt'

"""### Helper Functions"""

#####################################################################
################           HELPER FUNCTIONS           ###############
#####################################################################


def euclidean(x1, x2):
    '''
    Function to calculate euclidean distance between two points
    '''
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
        print(f'It took {hrs} hrs, {mins} mins and {secs} secs to run.')
    elif mins:
        print(f'It took {mins} mins and {secs} secs to run.')
    else:
        print(f'It took {secs} secs to run.')


def print_time(time_input):
    '''
    function to print the time with appropritate precision if between 0 and 1
    else print in HH, MM, SS format
    '''
    if time_input <= 1e-5:
        print(f'It took {time_input:.6f} secs to run.')
    elif time_input <= 1e-4:
        print(f'It took {time_input:.5f} secs to run.')
    elif time_input <= 1e-3:
        print(f'It took {time_input:.4f} secs to run.')
    elif time_input <= 1e-2:
        print(f'It took {time_input:.3f} secs to run.')
    elif time_input <= 1e-1:
        print(f'It took {time_input:.2f} secs to run.')
    elif time_input >= 0 and time_input <= 1:
        print(f'It took {time_input} secs to run.')
    else:
        print_formatted_time(time_input)

def add_line_break(lis_of_nums, break_every=2):
    '''
    function to split a list by adding \n to it.
    adds after every specified digit.
    Used for xticks values in graphs
    '''
    res = []
    for idx, i in enumerate(lis_of_nums):
        res.append(i)
        if idx != 0 and idx % break_every == 0:
            res.append('\n')
    return res

def plot_graphs(accuracy_map, algorithm = 'Forward Selection', dataset_size='Small'):
    '''
    Function to plot the feature graphs.
    Takes in accuracy_map array that is generated by the functions.
    '''
    plt.figure(1, figsize=(20, 5))
    plt.plot(accuracy_map[:, 0], accuracy_map[:, 1])
    plt.title(f'Accuracy vs Number of features for {algorithm} run on {dataset_size} Dataset')
    plt.xlabel('Selected features')
    plt.ylabel('Accuracy')
    plt.xticks(list(accuracy_map[:, 0]), list(
        [','.join(add_line_break([str(j) for j in i])) for i in accuracy_map[:, 2]]))
    plt.grid()
    plt.plot()
    plt.show()


def estimate_run_time(dataset_path, sep='  ', validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1):
    '''
    Function to check how many combinations of the Features the model needs to test and
    and how long will it take for the model to completely execute all the combinations
    '''

    df = pd.read_csv(f'{base_path}/{dataset_path}', sep=sep,
                     header=None, engine='python')
    print(
        f'The dataset {dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])

    if validation_type not in ['LEAVE_ONE_OUT', 'K_FOLD_VALIDATION']:
        raise Exception('Not a valid validation')

    k_fold_k_value = k_val
    if validation_type == 'LEAVE_ONE_OUT':
        k_fold_k_value = df.shape[0]

    if sampling_factor < 0 or sampling_factor > 1:
        raise Exception('Not a valid sampling factor')

    if sampling == True:
        # shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        selected_len = int(len(indices) * sampling_factor)
        X = X[:selected_len]
        Y = Y[:selected_len]
        k_fold_k_value = X.shape[0]


    t0 = time.time()
    k_fold_acc = k_fold_cross_validation(
        X, Y, k_fold_k_value, best_so_far=-1, tolerence=0, verbose=False)
    t1 = time.time()

    time_taken = t1 - t0
    total_combinations = (X.shape[1]*(X.shape[1]+1)) // 2

    print(
        f'K-fold cross validation accuracy on {dataset_path} for k = {k_fold_k_value} is with all features selected is {k_fold_acc*100:.3f}%')
    print_time(time_taken)
    print()
    print(
        f'The model will run for {total_combinations} times and will take a total time of {total_combinations*time_taken:.2f}s')
    print_time(total_combinations*time_taken)

"""### Main Functions"""

#####################################################################
################      NEAREST NEIGHBOR FUNCTION       ###############
#####################################################################


def knn(x_train, y_train, x_test, y_test):
    '''
    Function to find the accuracy of given train and test set using nearest neighbour algorithm
    '''
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


def k_fold_cross_validation(X, Y, k, best_so_far, tolerence=5, verbose=False, prune_run=False):
    '''
    Function to apply k fold cross validation on a given dataset
    This function can be optimized to terminate runs if the running average is lower than the best_so_far value provided.
    To terminate runs, pass the least required average, else pass in -1.
    The algorithm will run for a minimum of 100 iterations and then will tolerate atmost tolerence amount of runs before terminating.
    '''

    accuracy_scores = []
    fold_size = len(X) // k

    # shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]

    if verbose:
        print(f'Total Instances: {X.shape[0]}, Total Features: {X.shape[1]}')
        print(f'K: {k}, Fold Size: {fold_size}')

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

        # If the algorithm has run for 20 folds and the running average has not improved, then terminate the run
        if prune_run == True and len(accuracy_scores) > 20 and best_so_far != -1 and running_average < best_so_far:
            counter += 1
            if counter >= tolerence:
                return running_average

    # return average accuracy
    return np.mean(accuracy_scores)

#####################################################################
################           FORWARD SELECTION          ###############
#####################################################################


def forward_selection(dataset_path, sep='  ', validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1, prune_run=False):

    df = pd.read_csv(f'{base_path}/{dataset_path}', sep=sep,
                     header=None, engine='python')
    print(
        f'The dataset {dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])

    if validation_type not in ['LEAVE_ONE_OUT', 'K_FOLD_VALIDATION']:
        raise Exception('Not a valid validation')

    k_fold_k_value = k_val
    if validation_type == 'LEAVE_ONE_OUT':
        k_fold_k_value = df.shape[0]

    if sampling_factor < 0 or sampling_factor > 1:
        raise Exception('Not a valid sampling factor')

    if sampling == True:
        # shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        selected_len = int(len(indices) * sampling_factor)
        X = X[:selected_len]
        Y = Y[:selected_len]
        k_fold_k_value = X.shape[0]


    values, counts = np.unique(Y, return_counts=True)
    default_rate = max(counts)/sum(counts)
    print(f'Default Rate is: {default_rate*100:.2f}% \n')

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
            k_fold_acc = k_fold_cross_validation(
                X, Y, k_fold_k_value, best_feature_accuracy, tolerence=10, verbose=verbose, prune_run=prune_run)
            t1 = time.time()
            if verbose == True:
                print(f'Time for 1 feature {t1 - t0:.2f}s')
            verbose = False

            print(
                f'The accuracy for features {new_features} is {k_fold_acc*100:.2f}%')

            if k_fold_acc > best_feature_accuracy:
                best_feature_accuracy = k_fold_acc

            temp_acc_list.append((i, k_fold_acc))
            time_per_feature.append(t1-t0)

        temp_acc_list = sorted(temp_acc_list, reverse=True, key=lambda x: x[1])

        selected_features.append(temp_acc_list[0][0])
        accuracy_map.append(
            (len(selected_features), temp_acc_list[0][1], list(selected_features)))

        print()
        print(
            f'Max accuracy of {temp_acc_list[0][1]*100:.2f}% for feature {temp_acc_list[0][0]} and the feature list is {selected_features}')
        print(
            f'Average time for each feature: {(sum(time_per_feature)/len(time_per_feature)):.2f}s')
        print(f'Total time for each feature: {(sum(time_per_feature)):.2f}s')

        if(temp_acc_list[0][1] > best_so_far):
            best_so_far = temp_acc_list[0][1]
            best_features = list(selected_features)

        print()

    t11 = time.time()
    print_time(t11-t00)
    print('----------------------------------------')

    print(
        f'Best Accuracy is {best_so_far*100:.2f}% with features {best_features}')

    accuracy_map = np.array(accuracy_map,  dtype=object)
    return accuracy_map

#####################################################################
################         BACKWARD ELIMINATION         ###############
#####################################################################


def backward_elimination(dataset_path, sep='  ', validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1, prune_run=False):

    df = pd.read_csv(f'{base_path}/{dataset_path}', sep=sep,
                     header=None, engine='python')
    print(
        f'The dataset {dataset_path} has {df.shape[0]} instances with {df.shape[1] - 1} features')

    X = np.array(df[list(range(1, df.shape[1]))])
    Y = np.array(df[0])

    if validation_type not in ['LEAVE_ONE_OUT', 'K_FOLD_VALIDATION']:
        raise Exception('Not a valid validation')

    k_fold_k_value = k_val
    if validation_type == 'LEAVE_ONE_OUT':
        k_fold_k_value = df.shape[0]

    if sampling_factor < 0 or sampling_factor > 1:
        raise Exception('Not a valid sampling factor')

    if sampling == True:
        # shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        selected_len = int(len(indices) * sampling_factor)
        X = X[:selected_len]
        Y = Y[:selected_len]
        k_fold_k_value = X.shape[0]


    values, counts = np.unique(Y, return_counts=True)
    default_rate = max(counts)/sum(counts)
    print(f'Default Rate is: {default_rate*100:.2f}% \n')

    selected_features = list(range(1, df.shape[1]))  # Start with all features
    accuracy_map = []
    best_so_far = default_rate
    best_features = selected_features.copy()

    # get accuracy when all features are selected
    k_fold_acc = k_fold_cross_validation(
        X, Y, k_fold_k_value, -1, tolerence=0, verbose=False)
    accuracy_map.append(
        (len(selected_features), k_fold_acc, list(selected_features)))

    print(
        f'The accuracy for all features {selected_features} is {k_fold_acc*100:.2f}%')

    t00 = time.time()
    while len(selected_features) > 1:
        temp_acc_list = []
        time_per_feature = []
        verbose = True

        best_feature_accuracy = 0

        for i in selected_features:
            new_features = selected_features.copy()
            new_features.remove(i)

            X = np.array(df[list(new_features)])
            Y = np.array(df[0])

            t0 = time.time()
            k_fold_acc = k_fold_cross_validation(
                X, Y, k_fold_k_value, best_feature_accuracy, tolerence=10, verbose=verbose, prune_run=prune_run)
            t1 = time.time()
            if verbose:
                print(f'Time for 1 feature: {t1 - t0:.2f}s')
            verbose = False

            print(
                f'The accuracy after removing feature {i} and for features {new_features} is {k_fold_acc*100:.2f}%')

            if k_fold_acc > best_feature_accuracy:
                best_feature_accuracy = k_fold_acc

            temp_acc_list.append((i, k_fold_acc))
            time_per_feature.append(t1 - t0)

        temp_acc_list = sorted(temp_acc_list, reverse=True, key=lambda x: x[1])

        selected_features.remove(temp_acc_list[0][0])
        accuracy_map.append(
            (len(selected_features), temp_acc_list[0][1], list(selected_features)))

        print()
        print(
            f'Max accuracy of {temp_acc_list[0][1] * 100:.2f}% without feature {temp_acc_list[0][0]} and selected_features as {selected_features}')
        print(
            f'Average time for each feature: {sum(time_per_feature) / len(time_per_feature):.2f}s')
        print(f'Total time for each feature: {sum(time_per_feature):.2f}s')
        print()

        if(temp_acc_list[0][1] > best_so_far):
            best_so_far = temp_acc_list[0][1]
            best_features = list(selected_features)

    t11 = time.time()
    print_time(t11 - t00)
    print('----------------------------------------')

    print(
        f'Best Accuracy is {best_so_far*100:.2f}% with features {best_features}')
    accuracy_map = np.array(accuracy_map,  dtype=object)
    return accuracy_map

"""### Test Data"""

#####################################################################
# ################          COMBINE FUNCTION            ###############
# #####################################################################

# ################          TEST DATA 1            ###############


# dataset_path = 'CS170_small_Data__32.txt'
# print('------------------ Time Estimation ----------------------')
# estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
#                   k_val=2, sampling=False, sampling_factor=1)

# print()
# print('------------------ Forward Selection ----------------------')
# accuracy_map_forward_1 = forward_selection(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# plot_graphs(accuracy_map_forward_1)

# print()
# print('------------------ Backward Elimination ----------------------')
# accuracy_map_backward = backward_elimination(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# accuracy_map_updated_1 = np.array(accuracy_map_backward, dtype=object)
# accuracy_map_updated_1[:, 0] = len(
#     accuracy_map_updated_1) - accuracy_map_updated_1[:, 0]
# plot_graphs(accuracy_map_updated_1)

# ################          TEST DATA 2            ###############

# dataset_path = 'CS170_small_Data__33.txt'
# print('------------------ Time Estimation ----------------------')
# estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
#                   k_val=2, sampling=False, sampling_factor=1)

# print()
# print('------------------ Forward Selection ----------------------')
# accuracy_map_forward_2 = forward_selection(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# plot_graphs(accuracy_map_forward_2)

# print()
# print('------------------ Backward Elimination ----------------------')
# accuracy_map_backward = backward_elimination(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# accuracy_map_updated_2 = np.array(accuracy_map_backward, dtype=object)
# accuracy_map_updated_2[:, 0] = len(
#     accuracy_map_updated_2) - accuracy_map_updated_2[:, 0]
# plot_graphs(accuracy_map_updated_2)

# ################          TEST DATA 3            ###############

# dataset_path = 'CS170_large_Data__33.txt'
# print('------------------ Time Estimation ----------------------')
# estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
#                   k_val=2, sampling=False, sampling_factor=1)

# print()
# print('------------------ Forward Selection ----------------------')
# accuracy_map_forward_3 = forward_selection(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# plot_graphs(accuracy_map_forward_3)

# print()
# print('------------------ Backward Elimination ----------------------')
# accuracy_map_backward = backward_elimination(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# accuracy_map_updated_3 = np.array(accuracy_map_backward, dtype=object)
# accuracy_map_updated_3[:, 0] = len(
#     accuracy_map_updated_3) - accuracy_map_updated_3[:, 0]
# plot_graphs(accuracy_map_updated_3)

# ################          TEST DATA 4            ###############

# dataset_path = 'CS170_large_Data__33.txt'
# print('------------------ Time Estimation ----------------------')
# estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
#                   k_val=2, sampling=False, sampling_factor=1)

# print()
# print('------------------ Forward Selection ----------------------')
# accuracy_map_forward_4 = forward_selection(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# plot_graphs(accuracy_map_forward_4)

# print()
# print('------------------ Backward Elimination ----------------------')
# accuracy_map_backward = backward_elimination(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# accuracy_map_updated_4 = np.array(accuracy_map_backward, dtype=object)
# accuracy_map_updated_4[:, 0] = len(
#     accuracy_map_updated_4) - accuracy_map_updated_4[:, 0]
# plot_graphs(accuracy_map_updated_4)

"""### Selected Data"""

# #####################################################################
# ################       Working on selected Data       ###############
# #####################################################################

# ################          SELECTED DATA 1            ###############
# dataset_path = small_dataset_path
# estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
#                   k_val=2, sampling=False, sampling_factor=1)

# accuracy_map_forward_5 = forward_selection(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# plot_graphs(accuracy_map_forward_5, 'Forward Selection', 'Small')

# accuracy_map_backward = backward_elimination(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# accuracy_map_updated_5 = np.array(accuracy_map_backward, dtype=object)
# accuracy_map_updated_5[:, 0] = len(
#     accuracy_map_updated_5) - accuracy_map_updated_5[:, 0]
# plot_graphs(accuracy_map_updated_5, 'Backward Elimination', 'Small')

# ################          SELECTED DATA 6            ###############
# dataset_path = large_dataset_path
# estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
#                   k_val=2, sampling=False, sampling_factor=1)

# accuracy_map_forward_6 = forward_selection(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# plot_graphs(accuracy_map_forward_6, 'Forward Selection', 'Large')

# accuracy_map_backward = backward_elimination(
#     dataset_path, validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1)
# accuracy_map_updated_6 = np.array(accuracy_map_backward, dtype=object)
# accuracy_map_updated_6[:, 0] = len(
#     accuracy_map_updated_6) - accuracy_map_updated_6[:, 0]
# plot_graphs(accuracy_map_updated_6, 'Backward Elimination', 'Large')

################          SELECTED DATA 7            ###############
dataset_path = xxx_large_dataset_path
estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
                  k_val=2, sampling=False, sampling_factor=1)
print ()
estimate_run_time(dataset_path, validation_type='K_FOLD_VALIDATION',
                  k_val=2, sampling=False, sampling_factor=1)
print ()
estimate_run_time(dataset_path, validation_type='LEAVE_ONE_OUT',
                  k_val=2, sampling=True, sampling_factor=0.3)
print ()
estimate_run_time(dataset_path, validation_type='K_FOLD_VALIDATION',
                  k_val=2, sampling=True, sampling_factor=0.3)
print ()

# accuracy_map_forward_7 = forward_selection(
#     dataset_path, validation_type='K_FOLD_VALIDATION', k_val=2, sampling=True, sampling_factor=0.3)
# plot_graphs(accuracy_map_forward_7, 'Forward Selection', 'XXXLarge')

accuracy_map_backward = backward_elimination(
    dataset_path, validation_type='K_FOLD_VALIDATION', k_val=2, sampling=True, sampling_factor=0.3)
accuracy_map_updated_7 = np.array(accuracy_map_backward, dtype=object)
accuracy_map_updated_7[:, 0] = len(
    accuracy_map_updated_7) - accuracy_map_updated_7[:, 0]
plot_graphs(accuracy_map_updated_7, 'Backward Elimination', 'XXXLarge')




"""### UI"""

def main_block():
    print ('---- My Awesome Feature Search Algorithm ----\n')
    print ('Select the Dataset Size\n1. Small\n2. Large\n3. Extra Large')
    dataset_choice = int(input('Enter Choice.'))
    if dataset_choice not in [1, 2, 3]:
        os.system('cls')
        print('Please enter correct choice.\n')
        main_block()
        return

    print ('\nSelect the Algorithm\n1. Forward Selection\n2. Backward Elimination')
    algorithm_choice = int(input('Enter Choice.'))
    if algorithm_choice not in [1, 2]:
        os.system('cls')
        print('Please enter correct choice.\n')
        main_block()
        return

    datasets = [small_dataset_path, large_dataset_path, xxx_large_dataset_path]
    selected_dataset = datasets[dataset_choice-1]

    validation_type = 'LEAVE_ONE_OUT'
    k_val=2
    sampling=False
    sampling_factor=1

    if dataset_choice == 3:
        validation_type = 'K_FOLD_VALIDATION'
        k_val=2
        sampling=True
        sampling_factor=0.5

    if algorithm_choice == 1:
        estimate_run_time(selected_dataset, validation_type=validation_type, k_val=k_val, sampling=sampling, sampling_factor=sampling_factor)
        accuracy_map_forward_6 = forward_selection(selected_dataset, validation_type=validation_type, k_val=k_val, sampling=sampling, sampling_factor=sampling_factor)
        plot_graphs(accuracy_map_forward_6)
    else:
        estimate_run_time(selected_dataset, validation_type=validation_type, k_val=k_val, sampling=sampling, sampling_factor=sampling_factor)
        accuracy_map_backward = backward_elimination(selected_dataset, validation_type=validation_type, k_val=k_val, sampling=sampling, sampling_factor=sampling_factor)
        accuracy_map_updated_7 = np.array(accuracy_map_backward, dtype=object)
        accuracy_map_updated_7[:, 0] = len(accuracy_map_updated_7) - accuracy_map_updated_7[: ,0]
        plot_graphs(accuracy_map_updated_7)

main_block()

"""### Real World Dataset"""

# https://github.com/allisonhorst/palmerpenguins

real_world_path = 'realworld.csv'
data = sns.load_dataset("penguins") # load penguins dataset from seaborn
data = data.dropna() # drop samples with missing values (NaN)

columns_to_number_map = {}
number_to_columns_map = {}

for idx, col in enumerate(data.columns):
    columns_to_number_map[col] = idx
    number_to_columns_map[idx] = col

LE = LabelEncoder()
data['species'] = LE.fit_transform(data['species'])
data['island'] = LE.fit_transform(data['island'])
data['sex'] = LE.fit_transform(data['sex'])

data.to_csv(f'{base_path}/{real_world_path}',index=False, header=None,sep=' ')

estimate_run_time(real_world_path, sep=' ', validation_type='LEAVE_ONE_OUT',
                  k_val=2, sampling=False, sampling_factor=1)

accuracy_map_forward_real = forward_selection(
    real_world_path, sep=' ', validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1, prune_run = False)
print (number_to_columns_map)
plot_graphs(accuracy_map_forward_real, 'Forward Selection', 'Real World')

accuracy_map_backward = backward_elimination(
    real_world_path, sep=' ', validation_type='LEAVE_ONE_OUT', k_val=2, sampling=False, sampling_factor=1, prune_run = False)
accuracy_map_updated_real = np.array(accuracy_map_backward, dtype=object)
accuracy_map_updated_real[:, 0] = len(
    accuracy_map_updated_real) - accuracy_map_updated_real[:, 0]
print (number_to_columns_map)
plot_graphs(accuracy_map_updated_real, 'Backward Elimination', 'Real World')
