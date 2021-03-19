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

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#           output = 'mhplot.pdf'       #
#########################################

def main(args):

    input = args.input
    out = args.output
    original_features = int(args.original_features)

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    modalities = input_file.columns.drop('y')[original_features:]
    X = input_file.loc[:,modalities] #value of covariates

    values = pd.DataFrame(index=modalities,columns=['AUC','pval','qval'])

    # Calculate AUC, pval and qval

    for mod in modalities:
        val = metrics.roc_auc_score(y, X[mod])
        values.loc[mod,'AUC'] = round(max(val,1-val),3)

    nbr_features = len(values.index)

    for mod in values.index:
        values.loc[mod,'pval'] = wilcoxon(X.iloc[y[y==1].index][mod],X.iloc[y[y==0].index][mod], alternative='two-sided', correction=True)[1]

    values['qval'] = abs(multi.multipletests(values['pval'], method = 'fdr_bh')[1])

    values['pval'] = round(-np.log(values['pval'].astype(float))/np.log(10),3)
    values['qval'] = round(-np.log(values['qval'].astype(float))/np.log(10),3)

    features = values.index.str.replace(' ',',').values.astype(str)
    mod1 = [val[0] for val in np.char.split(features,'+')]
    mod1 = [val[0] for val in np.char.split(mod1,'*')]
    mod2 = [val[-1] for val in np.char.split(features,'+')]
    unique_mod = []
    for mod in mod1 :
        if mod not in unique_mod: unique_mod.append(mod)

    # Plot

    fig, axes = plt.subplots(nrows=3, ncols=1)
    df = pd.DataFrame(mod1, columns=['mod1'])
    colormap = ['orange', 'green', 'purple', 'blue', 'cyan']
    colors = dict(zip(unique_mod,colormap))
    Max = pd.DataFrame(index=unique_mod, columns=values.columns)
    counts = []
    ticks = []
    for i in range(len(unique_mod)):
        Max.loc[unique_mod[i]] = (values[values.index.str.startswith(unique_mod[i])].astype(float)).idxmax(axis=0)
        counts.append(mod1.count(unique_mod[i]))
        ticks.append(counts[i]/2 + sum(counts[:i]))

    for ax, value in zip(fig.axes, values.columns.values):
        plt.sca(ax)
        ax.scatter(values.index, values.loc[:,value], s=20, c=df['mod1'].map(colors))
        for i in Max.index:
            maxx = Max.loc[i,value]
            plt.annotate(maxx.split('+')[-1], (list(values.index.values).index(maxx), values.loc[maxx,value]))

        locs, labels = plt.xticks()
        x = [min(locs), max(locs)]
        mean = values[value].mean()
        y = [mean, mean]
        plt.plot(x, y, color="midnightblue", linewidth=1)

        plt.xticks(ticks, colors.keys(), rotation=90)
        plt.ylabel(values.columns[-int(round(3*ax.get_position().y0)+1)])

    # plt.show()
    plt.gcf().set_size_inches(15, 15)
    plt.savefig(out)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input csv file (original or interraction features)')
    parser.add_argument('--output','-o',default='out/mahplot.pdf',help='output filename')
    parser.add_argument('--original_features',default=0,help='number of original features without interractions')
    args = parser.parse_args()

    main(args)
