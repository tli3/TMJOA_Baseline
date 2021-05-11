import argparse
import csv
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#           output = 'ROC.pdf'          #
#########################################

def main(args):

    input = args.input
    folder = args.folder
    if args.output is None:
        out = args.folder+'/ROC.pdf'
    else :
        out = args.output

    print('Creating: ',os.path.basename(out))

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    modalities = input_file.columns.drop('y')
    X = input_file.loc[:,modalities] #value of covariates

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == 'Predictions.csv':
                results = pd.read_csv(os.path.join(root,file), index_col=0)
                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)
                for res in results.columns:
                    pred = results.loc[:,res]
                    fpr, tpr, _ = metrics.roc_curve(y, pred)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    aucs.append(metrics.auc(fpr, tpr))
                mean_tpr = np.mean(np.array(tprs), axis=0)
                mean_auc = round(metrics.auc(mean_fpr, mean_tpr),3)
                plt.plot(mean_fpr, mean_tpr,label=os.path.basename(root)+', auc='+str(mean_auc))

    importance = pd.read_csv(folder+'/FinalModel/Importance.txt', sep=",", header=None)
    features = importance[importance[1].cumsum() < 0.8].iloc[:,0]

    for feat in features[:min(len(features),8)]:
        curve_pred = X[feat]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label in [0,1]:
            fpr[label], tpr[label], _ = metrics.roc_curve(y, curve_pred, pos_label=label)
            roc_auc[label] = round(metrics.auc(fpr[label], tpr[label]),3)

        ind_max = max(roc_auc.items(), key=operator.itemgetter(1))[0]
        plt.plot(fpr[ind_max],tpr[ind_max], '--',label=feat.split('+')[-1]+', auc='+str(roc_auc[ind_max]), alpha=0.6)

    plt.plot([0,1],[0,1], c='grey', linewidth=1)
    plt.legend(loc=0)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(out)
    # plt.show()

    print('Saving: ',os.path.basename(out))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',default='interactions.csv',help='input interaction features csv file')
    parser.add_argument('--output','-o',help='output filename')
    parser.add_argument('--folder',default='Models',help='folder to evaluate')
    args = parser.parse_args()

    main(args)
