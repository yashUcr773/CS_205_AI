# Feature Selection using Nearest Neighbour.

Feature selection is a crucial step in machine learning, involving the identification of relevant and informative features from a dataset. Its importance lies in improving model performance, reducing overfitting, and enhancing interpretability. By selecting only the most influential factors, models achieve better accuracy, faster training times, and become more interpretable. Additionally, feature selection helps avoid overfitting and reduces storage and processing requirements, making models more feasible for deployment in resource-constrained environments.

This project aims to identify the most relevant and irrelevant features through forward selection, backward elimination, and nearest neighbor search techniques. The purpose of this undertaking is to fulfill the course requirement for CS-205: Artificial Intelligence at UCR. By employing these methods, the project seeks to enhance the understanding and performance of artificial intelligence algorithms in feature selection.

### Notes
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
