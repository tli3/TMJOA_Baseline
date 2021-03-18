import os
import csv
import math
import argparse
import numpy as np
import pandas as pd
from colour import Color
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn import metrics
import statsmodels.stats.multitest as multi
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#        output = 'Boxplot.pdf'         #
#########################################

def main(args):

    input = args.input
    folder = args.folder
    if args.output is None:
        out = args.folder

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    modalities = input_file.columns.drop('y')
    X = input_file.loc[:,modalities] #value of covariates

    top_importance = pd.read_csv(folder+'/Importance.txt', sep=",", header=None)
    features = list(top_importance[top_importance[1].cumsum() < 0.8].iloc[:,0])
    top_val = (X[features]-X[features].mean())/X[features].std()

    ##### Boxplot normalized values #####

    y = y.astype(bool)
    OA = top_val[y]
    OA['Label'] = 'OA'
    control = top_val[~y]
    control['Label'] = 'Control'
    values = pd.DataFrame(index=features,columns=['AUC'])

    # Calculate AUCs

    for feat in features:
        val = metrics.roc_auc_score(y, top_val[feat])
        values.loc[feat,'AUC'] = round(max(val,1-val),3)

    # Boxplot

    fig = plt.figure()
    plot_val = pd.concat([OA, control])
    box = pd.melt(plot_val, id_vars=['Label'], value_vars=features)
    flierprops = dict(marker='o', markersize=4, alpha=0.5, linestyle='none')
    bp = sns.boxplot(x='variable', y='value', data=box, hue='Label', palette=['orangered','lightskyblue'], flierprops=flierprops)

    # Axis and legends

    bp.set_xlabel('')
    bp.set_ylabel('Normalized measures', size=12)
    bp.grid('On')

    features_name = [val[-1] for val in np.char.split(features,'+')]
    AUC = list(values.loc[:,'AUC'])
    label = [feat + '(AUC: ' + str(auc) + ')' for feat,auc in zip(features_name,AUC)]
    bp.set_xticklabels(label,rotation=90)

    fig.subplots_adjust(bottom=0.4, top=0.95)
    plt.title('Boxplots of top features for OA vs Control', fontweight='bold')
    plt.legend(loc=0)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(out+'/Boxplot_values.pdf')
    # plt.show()
    plt.close()

    ##### Boxplot features contribution #####

    importance = pd.read_csv(folder+'/Importance.csv', index_col=0)
    contribution = importance.loc[features].transpose()

    fig = plt.figure()
    flierprops = dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5, linestyle='none')
    meanprops = dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=5)
    bp = sns.boxplot(data=contribution, notch=True, orient='h', flierprops=flierprops, showmeans=True, meanprops=meanprops)

    # Axis and legends
    fig.subplots_adjust(left=0.25)
    bp.set_xlabel('Model contribution', size=12)
    bp.set_yticklabels(features_name)
    plt.title('Boxplots of top '+str(len(features))+' features (>80%) in '+folder, fontweight='bold')
    bp.grid('On')
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(out+'/Boxplot_contribution.pdf')
    # plt.show()
    plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',default='interractions.csv',help='input interraction features csv file')
    parser.add_argument('--output','-o',help='output folder')
    parser.add_argument('--folder','-f',default='XGBoost/eta0.01W1C0.5S0.5',help='Folder to evaluate')
    args = parser.parse_args()

    main(args)
