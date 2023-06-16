Feature Selection using Nearest Neighbour.

Project to find the most relevent and irrelevent features using forward selection and backward elimination and nearest neighbout search
Completed For CS-205: Artificial Intelligence at UCR.

- The code is menu driven and can be run directly without changes.
- The code is extensible and can be run for any synthetic dataset, just need to enter call the function `run_test_data_1` and update the variable`dataset_path` with new dataset.
- The code can run any of the test datasets using the functions `run_test_data_1`, `run_test_data_2`, `run_test_data_3` and `run_test_data_4`. By defaultm this is commented out.
- The code can run the assigned datasets using the functions `run_selected_data_1`, `run_selected_data_2` and `run_selected_data_3`. By defaultm this is commented out.
- The code can run the real world dataset using the function `real_world`. By defaultm this is commented out.
- The code can run any of the assigned dataset for any algorithm using the menu. By default this is the only function that is called.
- All the functions run both forward selection and backward elimination but can be controlled using flags. `run_test_data_1(forward = True, backward = False)`
- All the functions generate graphs as required.
- The k-fold validation function has pruning built in to halt runs in case the accuracy is less than desired.

Colab Link --> https://colab.research.google.com/drive/1VyzhBqGJSqQLU3EI7kodLMSerO-ZY5Us
