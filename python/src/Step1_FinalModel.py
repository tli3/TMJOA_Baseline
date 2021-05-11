import argparse
import csv
import os
import pickle
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#       output = 'Predictions.csv'      #
#########################################


def main(args):
    
    print('Prediction in progress...')

    interactions = args.interactions
    auc = args.auc
    out = args.output
    folder = args.folder

    if out[-1]!='/':
        out=out+'/'
    if folder[-1]!='/':
        folder=folder+'/'

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass

    interactions = pd.read_csv(interactions)
    AUC = pd.read_csv(auc, index_col=0)
    seed1 = int(AUC.columns[0].split('_')[0])
    seed_end = int(AUC.columns[-1].split('_')[0]) + 1
    nbr_seed = seed_end-seed1
    nbr_folds = int(AUC.columns[-1].split('_')[-1])

    y = interactions['y'] #result(0 or 1)
    interactions = interactions[interactions.columns.drop(['y'])]
    
    mean_stat = pd.DataFrame(0, index=['mean'], columns=['ACC','PREC1','PREC0','RECALL1','RECALL0','F1SCORE','AUC'])
    mean_importance = pd.DataFrame()

    # Prediction
    modelNames = ['XGBoost','LightGBM']
    seeds = range(seed1,seed_end)
    folds = range(nbr_folds)
    mean_pred = pd.DataFrame(0, index=range(len(y)), columns=['Pred'])
    nbr_models = len(modelNames)*len(seeds)
    for modelName in modelNames:
        for seed in seeds:
            pred = pd.read_csv(folder+modelName+'/Predictions.csv', index_col=0)[[str(seed)]]
            mean_pred = mean_pred.add(pred.values)
            for fold in folds:
                importance = pd.read_csv(folder+modelName+'/Importance.csv', index_col=0)[str(seed)+'_'+str(fold+1)]
                mean_importance[modelName+'_'+str(seed)+'_'+str(fold+1)] = importance

    mean_pred['Pred'] = [x/nbr_models for x in mean_pred['Pred']]
    # print(mean_pred)
    mean_pred.to_csv(out+'Predictions.csv')

    # Results
    stat = pd.DataFrame(index=['mean'],columns=['ACC','PREC1','PREC0','RECALL1','RECALL0','F1SCORE','AUC'])
    pred_bool = np.array(mean_pred['Pred']).astype(float).round(0).astype(bool)
    y_bool = y.astype(bool)
    acc = round(metrics.accuracy_score(y_bool, pred_bool),4)
    prec1 = round(metrics.precision_score(y_bool, pred_bool),4)
    prec0 = round(metrics.precision_score(~y_bool, ~pred_bool),4)
    recall1 = round(metrics.recall_score(y_bool, pred_bool),4)
    recall0 = round(metrics.recall_score(~y_bool, ~pred_bool),4)
    f1 = round(metrics.f1_score(y_bool, pred_bool),4)
    auc = round(metrics.roc_auc_score(y, mean_pred['Pred']),4)
    stat.loc['mean'] = [acc,prec1,prec0,recall1,recall0,f1,auc]
    # stat = stat.round(decimals=4)
    print(stat)

    mean = mean_importance.mean(axis=1)
    mean_importancetxt = pd.DataFrame(mean, index = mean.index, columns = ['mean'])
    mean_importancetxt = mean_importancetxt.sort_values(by=['mean'], ascending=False)
    mean_importance.to_csv(out+'Importance.csv')
    stat.to_csv(out+'Stat.csv', index=False)
    mean_importancetxt.to_csv(out+'Importance.txt', header=False)

    print('Saving: Final model')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions','-i',default='interactions.csv',help='input csv interraction file to test')
    parser.add_argument('--auc',default='AUC.csv',help='input csv AUC file')
    parser.add_argument('--output','-o',default='Models/FinalModel/',help='output folder')
    parser.add_argument('--folder','-f',default='Models/',help='models folder')
    args = parser.parse_args()

    main(args)
