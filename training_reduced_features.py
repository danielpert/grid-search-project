# train random forest regressors using the highest-scoring parameters found
# with the signac grid search and use the features with the top 75%, 50%, and
# 25% feature importances based on the feature importances from the highest
# scoring models. Also create json files for each model with the values of COF
# and intercept predicted by the model and the values from simulation for plotting

#import relevant packages
import numpy as np
import pandas as pd
import sklearn
import json
import signac
import pickle
from sklearn import ensemble

from mcl import input_prep

# Define global variables
TARGETS = ['COF', 'intercept']
IDENTIFIERS = ['terminal_group_1', 'terminal_group_2', 'terminal_group_3',
               'backbone', 'frac-1', 'frac-2']

# Declare SMILES dict to use for calculating features
smiles_dict = {'acetyl': ['C(=O)C', 'CC(=O)C'],
 'amino': ['N', 'CN'],
 'carboxyl': ['C(=O)O', 'CC(=O)O'],
 'cyano': ['C#N', 'CC#N'],
 'cyclopropyl': ['C1CC1', 'CC1CC1'],
 'difluoromethyl': ['FCF', 'FC(F)C'],
 'ethylene': ['C=C', 'CC=C'],
 'fluorophenyl': ['C1=CC=C(F)C=C1', 'CC1=CC=C(F)C=C1'],
 'hydroxyl': ['O', 'CO'],
 'isopropyl': ['C(C)C', 'CC(C)C'],
 'methoxy': ['OC', 'COC'],
 'methyl': ['C', 'CC'],
 'nitro': ['[N+](=O)[O-]', 'C[N+](=O)[O-]'],
 'nitrophenyl': ['C1=CC=C([N+](=O)[O-])C=C1', 'CC1=CC=C([N+](=O)[O-])C=C1'],
 'perfluoromethyl': ['C(F)(F)F', 'CC(F)(F)F'],
 'phenol': ['c1ccc(cc1)O', 'Cc1ccc(cc1)O'],
 'phenyl': ['C1=CC=CC=C1', 'CC1=CC=CC=C1'],
 'pyrrole': ['C1=CNC=C1', 'C1=C(C)NC=C1'],
 'toluene': ['Cc1ccccc1', 'Cc1ccc(C)cc1']}

path_to_desc = 'csv-files/descriptors-ind.csv'
        

if __name__ == '__main__':
    
    # import highest scoring models from pickle files
    with open('trained-models/best_COF_trained.pickle', 'rb') as f:
        COF_model = pickle.load(f)
    with open('trained-models/best_intercept_trained.pickle', 'rb') as f:
        intercept_model = pickle.load(f)
    # import list of features included in highest scoring models from json
    with open('json-files/best_features.json', 'r') as f:
        best_features = json.load(f)
        
    # create feature importances dictionary
    feature_importances = {'COF': (best_features['COF'], COF_model.feature_importances_.tolist()), \
                           'intercept': (best_features['intercept'], intercept_model.feature_importances_.tolist())}
    
    # create pandas dataframes of feature importances to aid in selection
    # and store them in a dictionary
    feature_importances_df = dict()
    feature_importances_df['COF'] = pd.DataFrame(np.array(feature_importances['COF'][1]), \
                                                 index=np.array(feature_importances['COF'][0]), \
                                                 columns=['importance'])
    feature_importances_df['intercept'] = pd.DataFrame(np.array(feature_importances['intercept'][1]), \
                                                       index=np.array(feature_importances['intercept'][0]), \
                                                       columns=['importance'])
    
    # reduce number of features based on importance
    # reduce by 25%, 50%, and 75%, to train models 1, 2, and 3 respectively
    reduced_features = dict()
    for target in TARGETS:
        for i, percent in enumerate([0.25, 0.5, 0.75]):
            percentile = feature_importances_df[target]['importance'].quantile(percent)
            reduced_features['{}_{}'.format(target, i+1)] = \
                feature_importances_df[target][feature_importances_df[target]['importance'] > percentile].index.tolist()
    
    # store reduced features in json file
    with open('json-files/reduced_features.json', 'w') as f:
        json.dump(reduced_features, f)
    
    # fetch best jobs
    project = signac.get_project()
    best_job = dict()
    best_job['COF'] = project.open_job(id='21a755aa4b3ae6d2556efdf563939ea5')
    best_job['intercept'] = project.open_job(id='01f2259c8b8ded041bcca2c3873e4be5')
    
    # iterate thru targets to train new models
    for target in TARGETS:
        
        # get cooresponding job for target to get statepoints for training
        job = best_job[target]
        
        # read training data
        path2traindata = 'csv-files/{}_training_1.csv'.format(target)
        training_df = pd.read_csv(path2traindata, index_col=0)

        # fetch test data for creating json files
        path_to_test = 'csv-files/{}_testing.csv'.format(target)
        with open(path_to_test, 'r') as test_file:
            test_df= pd.read_csv(test_file, index_col=0)
        test_df.sort_index(axis=0, inplace=True)
        numrows = test_df.shape[0]
        
        # iterate thru percentiles of features to remove to train new models
        # 1 removes bottom 25% of features by importance
        # 2 removes bottom 50% "
        # 3 removes bottom 75% "
        for i in range(1, 4):
            
            # split into X and y
            X_train, y_train = (training_df[reduced_features['{}_{}'.format(target, i)]], training_df[target])
            
            # create regressor object and train on data
            regr = ensemble.RandomForestRegressor(n_estimators=1000,
                                              oob_score=True,
                                              max_features=job.sp.max_features,
                                              max_depth=job.sp.max_depth,
                                              min_samples_split=job.sp.min_samples_split,
                                              min_samples_leaf=job.sp.min_samples_leaf,
                                              random_state=43)
            regr.fit(X_train, y_train)
               
            # pickle out
            with open('trained-models/{}_{}_trained.pickle'.format(target, i), 'wb') as to_write:
                pickle.dump(regr, to_write)
            print('Pickling out to {}_{}_trained.pickle'.format(target, i))
            
            ###
            ### create predicted json file for model from test data
            ###
            
            results = dict()
        
            for idx, row in test_df.iterrows():
                SMILES1 = smiles_dict[row['terminal_group_1']][0]
                SMILES2 = smiles_dict[row['terminal_group_2']][0]
                SMILES3 = smiles_dict[row['terminal_group_3']][0]
                frac1 = row['frac-1']
                frac2 = row['frac-2']

                predict_df = input_prep(terminal_groups = [SMILES1, SMILES2,
                                        SMILES3],
                                        fractions = [frac1, frac2],
                                        features_to_include = reduced_features['{}_{}'.format(target, i)],
                                        path_to_desc = path_to_desc)
                predicted = \
                    regr.predict(np.asarray(predict_df).reshape(1,-1))
                results[idx] = {'tg-1': row['terminal_group_1'],
                                'frac-1': row['frac-1'],
                                'tg-2': row['terminal_group_2'],
                                'frac-2': row['frac-2'],
                                'tg-3': row['terminal_group_3'],
                                'predicted-{}'.format(target): predicted[0],
                                'simulated-{}'.format(target): row[target]}
                print('Added data for row {} out of {} for {} model, {}% complete'.format(
                    idx, numrows-1, target, 100.0*(idx+1)/numrows))
            print('{} rows in json file for {}_{}_predicted'.format(len(results), target, i))
            with open('json-files/{}_{}_predicted.json'.format(target, i), 'w') as f:
                json.dump(results, f)
                