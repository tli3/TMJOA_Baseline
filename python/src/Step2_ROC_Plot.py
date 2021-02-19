import os
import csv
import argparse
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#           output = 'ROC.pdf'          #
#########################################

def main(args):

    input = args.input
    out = args.output
    xgb = args.xgboost
    lgb = args.lightgbm

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    modalities = input_file.columns.drop('y')
    X = input_file.loc[:,modalities] #value of covariates

    xgb_results = pd.read_csv(xgb+'/Pred.csv')
    lgb_results = pd.read_csv(lgb+'/Pred.csv')
    curve_names = ['XGBoost','LightGBM','XGBoost+LightGBM']

    pred_xgb = xgb_results.mean(axis=1)
    pred_lgb = lgb_results.mean(axis=1)
    pred_xgb_lgb = (pred_xgb+pred_lgb)/2
    curve_preds = [pred_xgb,pred_lgb,pred_xgb_lgb]

    for curve_name,curve_pred in zip(curve_names,curve_preds):
        fpr,tpr,thresh = metrics.roc_curve(y, curve_pred)
        auc = round(metrics.auc(fpr, tpr),2)
        plt.plot(fpr,tpr,label=curve_name+', auc='+str(auc))

    xgb_importance = pd.read_csv(lgb+'/Importance.txt', sep=",", header=None)
    features = xgb_importance[xgb_importance[1].cumsum() < 0.8].iloc[:,0]

    for feat in features[:max(len(features),8)]:
        curve_pred = X[feat]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label in [0,1]:
            fpr[label], tpr[label], _ = metrics.roc_curve(y, curve_pred, pos_label=label)
            roc_auc[label] = round(metrics.auc(fpr[label], tpr[label]),3)

        ind_max = max(roc_auc.items(), key=operator.itemgetter(1))[0]
        plt.plot(fpr[ind_max],tpr[ind_max], '--',label=feat.split('+')[-1]+', auc='+str(roc_auc[ind_max]))

    plt.plot([0,1],[0,1], c='grey', linewidth=1)
    plt.legend(loc=0)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(out)
    plt.show()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',default='interractions.csv',help='input interraction features csv file')
    parser.add_argument('--output','-o',default='out/ROC.pdf',help='output filename')
    parser.add_argument('--xgboost','-xgb',default='XGBoost/eta0.01W1C0.5S0.5',help='XGBoost folder')
    parser.add_argument('--lightgbm','-lgb',default='LightGBM/eta0.01W1C0.5S0.5',help='LightGBM folder')
    args = parser.parse_args()

    main(args)
