# Train random forest regressor model for specific job and
# pickle models out to {TARGET}_trained.pickle 

# Import relevant packages
import signac
from flow import FlowProject
from collections import OrderedDict
import json
import os
import numpy as np 
import pandas as pd
import scipy
from sklearn import ensemble, linear_model, metrics, model_selection
import pickle
import atools_ml

from atools_ml.dataio import df_setup
from atools_ml.prep import dimensionality_reduction, train_test_split

# Define global variables
TARGETS = ['COF', 'intercept']
IDENTIFIERS = ['terminal_group_1', 'terminal_group_2', 'terminal_group_3',
               'backbone', 'frac-1', 'frac-2']

# define group that combines the different operations
random_forest = FlowProject.make_group(name='random_forest')

@FlowProject.label
def models_exist(job):
    # post condition for train_models
    # return true if pickle files for models are in job workspace, else false
    
    for target in TARGETS:
        if not os.path.isfile(job.fn('{}_trained.pickle'.format(target))):
            return False
    return True


@FlowProject.label
def features_in_doc(job):
    # post condition for train_models
    # return true if job document contains model features for both COF and
    #intercept, else false
    
    return all([job.document.get('{}_features'.format(target)) for target in TARGETS])


@FlowProject.label
def scores_in_doc(job):
    # post condition for score
    # return true if job document contains scores for both COF and
    #intercept, else false
    
    return all([job.document.get('{}_score'.format(target)) for target in TARGETS])


@random_forest
@FlowProject.operation
@FlowProject.post(models_exist)
@FlowProject.post(features_in_doc)
def train_models(job):
    '''
    given a job with specific paramters, train a random forest
    regressor with said parameters and pickle the model to a 
    file inside the job's workspace
    '''
    
    for target in TARGETS:
        
        # read training data
        path2traindata = 'csv-files/{}_training_1.csv'.format(target)
        training_df = pd.read_csv(path2traindata, index_col=0)
        
        # Reduce the number of features by running data thru dimensionality reduction
        features = list(training_df.drop(TARGETS + IDENTIFIERS, axis=1))
        print('features before dimensionality reduction')
        print(features)
        df_red_train = dimensionality_reduction(training_df, features,
                                                missing_threshold=0.4,
                                                var_threshold=job.sp.var_threshold,
                                                corr_threshold=job.sp.corr_threshold)
        df_train = df_red_train
        features = list(df_train.drop(TARGETS + IDENTIFIERS, axis=1))
        print('features after dimensionality reduction')
        print(features)
        
        # split into X and y
        X_train, y_train = (df_red_train[features], df_red_train[target])
        
        # create regressor object and train on data
        regr = ensemble.RandomForestRegressor(n_estimators=1000,
                                              oob_score=True,
                                              max_features=job.sp.max_features,
                                              max_depth=job.sp.max_depth,
                                              min_samples_split=job.sp.min_samples_split,
                                              min_samples_leaf=job.sp.min_samples_leaf,
                                              random_state=43)
        print('X train')
        print(X_train)
        print('y train')
        print(y_train)
        regr.fit(X_train, y_train)
        
        with open(job.fn('{}_trained.pickle'.format(target)), 'wb') as to_write:
            pickle.dump(regr, to_write)
        print('Pickling out to {}_trained.pickle'.format(target))
        
        # store features data in job document
        job.doc['{}_features'.format(target)] = features 


@random_forest
@FlowProject.operation
@FlowProject.pre(models_exist)
@FlowProject.post(scores_in_doc)
def score(job):
    '''
    scores the COF and intercept models trained
    for a particular job (or combination of parameters).
    adds scores to job document
    '''
    
    for target in TARGETS:
        
        # fetch model
        with open(job.fn('{}_trained.pickle'.format(target)), 'rb') as modelfile:
            model = pickle.load(modelfile)
            
        # fetch test data
        path2testdata = 'csv-files/{}_testing.csv'.format(target)
        with open(path2testdata, 'r') as test_file:
            test_df= pd.read_csv(test_file, index_col=0)
        
        # split test data into features and target
        features = job.doc['{}_features'.format(target)]
        test_X = test_df[features]
        test_y = test_df[target]
        
        # score model and add to job document
        r2 = model.score(test_X, test_y)
        job.doc['{}_score'.format(target)] = r2
        
        
    
if __name__ == '__main__':
    FlowProject().main()