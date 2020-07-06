# Create json files for best models

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

IDENTIFIERS = ['terminal_group_1', 'terminal_group_2', 'terminal_group_3',
               'backbone', 'frac-1', 'frac-2']


def input_prep(terminal_groups, fractions,
               features_to_include,
               path_to_desc='csv-files/descriptors-ind.csv'):
    """
    Turning SMILES strings and fractions into pd.Series of feature values
    given the features to include that can be fed into machine learning model
    
    Parameters
    ----------
    terminal_groups : list of int
        List of terminal groups, bot-bot-top, len of 3 
    fractions : list of int
        Fraction of bot1 and bot 2, len of 2
    features_to_include : list
        list of features to include in dataframe
    """
    SMILES1 = terminal_groups[0]
    frac1 = fractions[0]
    SMILES2 = terminal_groups[1]
    frac2 = fractions[1]
    SMILES3 = terminal_groups[2]
    random_seed = 43

    to_drop = ['pc+-mean', 'pc+-min', 'pc--mean', 'pc--min']
    h_ch3_conv = {'C(=O)C': 'CC(=O)C',
                  'N': 'CN',
                  'C(=O)O': 'CC(=O)O',
                  'C#N': 'CC#N',
                  'C1CC1': 'CC1CC1',
                  'FCF': 'FC(F)C',
                  'C=C': 'CC=C',
                  'C1=CC=C(F)C=C1': 'CC1=CC=C(F)C=C1',
                  'O': 'CO',
                  'C(C)C': 'CC(C)C',
                  'OC': 'COC',
                  'C': 'CC',
                  '[N+](=O)[O-]': 'C[N+](=O)[O-]',
                  'C1=CC=C([N+](=O)[O-])C=C1': 'CC1=CC=C([N+](=O)[O-])C=C1',
                  'C(F)(F)F': 'CC(F)(F)F',
                  'c1ccc(cc1)O': 'Cc1ccc(cc1)O',
                  'C1=CC=CC=C1': 'CC1=CC=CC=C1',
                  'C1=CNC=C1': 'C1=C(C)NC=C1',
                  'Cc1ccccc1': 'Cc1ccc(C)cc1'}
    ch3_SMILES1 = h_ch3_conv[SMILES1]
    ch3_SMILES2 = h_ch3_conv[SMILES2]
    ch3_SMILES3 = h_ch3_conv[SMILES3]

    with open('json-files/feature-clusters.json', 'r') as f:
        clusters = json.load(f) # this is a dict
    shape_features = clusters['shape'] # a list from the clusters dict

    raw_desc_df = pd.read_csv(path_to_desc, index_col=0)
    raw_desc_dict = raw_desc_df.to_dict()
    
    # Descriptors for H-terminated SMILES
    desc_h_tg1 = raw_desc_dict[SMILES1]
    desc_h_tg2 = raw_desc_dict[SMILES2]
    desc_h_tg3 = raw_desc_dict[SMILES3]

    # Descriptors for CH3-terminated SMILES
    desc_ch3_tg1 = raw_desc_dict[ch3_SMILES1]
    desc_ch3_tg2 = raw_desc_dict[ch3_SMILES2]
    desc_ch3_tg3 = raw_desc_dict[ch3_SMILES3]
    
    
    desc_h_combined = dict()
    for key in desc_h_tg1:
        desc_h_combined[key] = desc_h_tg1[key]*frac1 + desc_h_tg2[key]*frac2

    desc_ch3_combined = dict() 
    for key in desc_ch3_tg1:
        desc_ch3_combined[key] = desc_ch3_tg1[key]*frac1 + desc_ch3_tg2[key]*frac2
        
    desc_h_df = pd.DataFrame([desc_h_combined, desc_h_tg3])
    desc_ch3_df = pd.DataFrame([desc_ch3_combined, desc_ch3_tg3]) 
        
        
    desc_df = []
    for i, df in enumerate([desc_h_df, desc_ch3_df]):
        if i == 1:
            hbond_tb = max(df['hdonors'][0], df['hacceptors'][1]) \
                       if all((df['hdonors'][0], df['hacceptors'][1])) \
                       else 0
            hbond_bt = max(df['hdonors'][1], df['hacceptors'][0]) \
                       if all((df['hdonors'][1], df['hacceptors'][0])) \
                       else 0
            hbonds = hbond_tb + hbond_bt
            df.drop(['hdonors', 'hacceptors'], 'columns', inplace=True)
        else:
            hbonds = 0
        means = df.mean()
        mins = df.min()
        means = means.rename({label: '{}-mean'.format(label)
                              for label in means.index})
        mins = mins.rename({label: '{}-min'.format(label)
                            for label in mins.index})
        desc_tmp = pd.concat([means, mins])
        desc_tmp['hbonds'] = hbonds
        desc_tmp.drop(labels=to_drop, inplace=True)
        desc_df.append(desc_tmp)
        
    df_h_predict = desc_df[0]
    df_ch3_predict = desc_df[1]
    df_h_predict = pd.concat([
        df_h_predict.filter(like=feature) for feature in shape_features], axis=0)
    df_ch3_predict.drop(labels=df_h_predict.keys(), inplace=True)

    df_h_predict_mean = df_h_predict.filter(like='-mean')
    df_h_predict_min = df_h_predict.filter(like='-min')
    df_ch3_predict_mean = df_ch3_predict.filter(like='-mean')
    df_ch3_predict_min = df_ch3_predict.filter(like='-min')

    df_predict = pd.concat([df_h_predict_mean, df_h_predict_min,
                            df_ch3_predict_mean, df_ch3_predict_min,
                            df_ch3_predict[['hbonds']]])
    
    return df_predict.filter(features_to_include)


def main():
    
    # Declare SMILES dict
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
    
    with open('json-files/best_features.json', 'r') as f:
        best_features = json.load(f)
    
    for target in ['COF', 'intercept']:
        print('Creating json files to store {} data'.format(target))
        path_to_test = 'csv-files/{}_testing.csv'.format(target)
        
        with open(path_to_test, 'r') as test_file:
            test_df= pd.read_csv(test_file, index_col=0)
        test_df.sort_index(axis=0, inplace=True)
        numrows = test_df.shape[0]
        
        
        path_to_model = 'trained-models/best_{}_trained.pickle'.format(target)
        
        with open(path_to_model, 'rb') as model_file:
            model = pickle.load(model_file)
        
        results = dict()
        
        for idx, row in test_df.iterrows():
            SMILES1 = smiles_dict[row['terminal_group_1']][0]
            SMILES2 = smiles_dict[row['terminal_group_2']][0]
            SMILES3 = smiles_dict[row['terminal_group_3']][0]
            frac1 = row['frac-1']
            frac2 = row['frac-2']

        #Run the use the loaded model to run the predict and save them out to dictionary -> csv files 
            predict_df = input_prep(terminal_groups = [SMILES1, SMILES2,
                                     SMILES3],
                                     fractions = [frac1, frac2],
                                     features_to_include = best_features[target],
                                     path_to_desc = path_to_desc)
            predicted = \
                model.predict(np.asarray(predict_df).reshape(1,-1))
            results[idx] = {'tg-1': row['terminal_group_1'],
                            'frac-1': row['frac-1'],
                            'tg-2': row['terminal_group_2'],
                            'frac-2': row['frac-2'],
                            'tg-3': row['terminal_group_3'],
                            'predicted-{}'.format(target): predicted[0],
                            'simulated-{}'.format(target): row[target]}
            print('Added data for row {} out of {} for {} model, {}% complete'.format(
                idx, numrows-1, target, 100.0*(idx+1)/numrows))
        print('{} rows in json file for {}_predicted'.format(len(results), target))
        with open('json-files/{}_predicted.json'.format(target), 'w') as f:
            json.dump(results, f)
            
                
if __name__ == '__main__':
    main()
            
