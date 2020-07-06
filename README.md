Signac grid search of random forest paramters to evaluate on test data instead of using cross validation on train data
Goal: reduce overfitting in selected model by scoring models on their ability to generalize to data not part of the training set

RandomForestRegressor parameters to search:
    max_features = ['log2', 'sqrt']
    max_depth = [None, 20, 40, 60, 80]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 5]
DimensionalityReduction parameters to search:
    var_threshold = [0.01, 0.02, 0.03]
    corr_threshold = [0.8, 0.9, 0.95]
*note: 810 total combinations
    
Each model has a unique combination of the parameters above and a unique ID assigned by signac, which is the directory name for the job.

The init.py file initializes the signac project and the training.py file has a random_forest group with two operations:
- `train_models` trains a COF and intercept random forest regressor model for each job, where each job represents a different combination of the parameters above for the random forest model and the dimensionality reduction. The trained models are pickled into the job workspace, and the model features are stored in the job document.
- `score` loads the pickle files and the model features from the job doc and finds the r^2 score for both the COF and intercept RF regressors, storing them in the job doc.

ModelEvaluation.ipynb is an interactive Jupyter notebook that loops through each of the jobs to create a pandas DataFrame, with each row representing a different job and the columns storing the parameters or signac statepoints, the number of features in the COF and intercept models, and the scores of the COF and intercept random forest models that were trained using those parameters. The parameters are all converted to numerical form, specifically to a float, in order to avoid DataFrame errors that prevented using the DataFrame.max() method. Additionally, the job ID was not included for this reason, however, the idx2id dictionary allows one to convert the index from the DataFrame to the job ID. In order to convert max_features to a numerical value, the square root or base 2 logarithm were taken of the number of features of each of the models when max_features was 'sqrt' or 'log2', respectively. A max_depth of None was converted to a numerical value of 2000 to represent the approximate theoretical maximum depth a tree could possibly be on a training size of 2088. The job ID, statepoints, and r^2 score were calculated for both the COF and intercept regressor susing this data frame, the best models were pickled out to a file in the home directory for the signac project, and the features included in the models with the highest score are stored in best_features.json. ModelEvaluation.ipynb also includes a histogram showing the score distributions for the COF and intercept models.

mcl.py contains code that uses the identifier information about the terminal groups and fractions, the input_prep function, and the descriptors-ind.csv file to create json files describing the predicted and simulated values of the target variables, and Plotting.ipynb plots the feature importances bar chart and a scatter plot showing the predicted and simulated values of the target variables.
