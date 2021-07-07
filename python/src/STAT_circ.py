import argparse
import csv
import math
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multi
from colour import Color
from scipy.stats import mannwhitneyu
from sklearn import metrics

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#           output = 'circ.pdf'         #
#########################################

def main(args):

    input = args.input
    out = args.output
    sort = args.sort
    original_features = int(args.original_features)
    # min_auc = float(args.min_auc)

    print('Creating: ',os.path.basename(out))

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    modalities = input_file.columns.drop('y')#[original_features:]
    X = input_file.loc[:,modalities] #value of covariates

    values = pd.DataFrame(index=modalities,columns=['AUC','pval','qval'])

    # Calculate AUC, pval and qval

    for mod in modalities:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label in [0,1]:
            fpr[label], tpr[label], _ = metrics.roc_curve(y, X[mod], pos_label=label)
            roc_auc[label] = round(metrics.auc(fpr[label], tpr[label]),3)
        ind_max = max(roc_auc.items(), key=operator.itemgetter(1))[0]
        values.loc[mod,'AUC'] = round(roc_auc[ind_max],3)
        values.loc[mod,'pval'] = mannwhitneyu(X.iloc[y[y==1].index][mod],X.iloc[y[y==0].index][mod], alternative='two-sided')[1]
    
    values['qval'] = abs(multi.multipletests(values['pval'], method = 'fdr_bh')[1])
    values['pval'] = round(-np.log(values['pval'].astype(float))/np.log(10),3)
    values['qval'] = round(-np.log(values['qval'].astype(float))/np.log(10),3)

    values = values.iloc[original_features:]
    values = values.sort_values(by=[sort], ascending=False)
    if args.nbr_features:
        nbr_features = args.nbr_features
    else:
        nbr_features = 30
    values = values.iloc[:nbr_features]

    # values = values.iloc[original_features:]
    # values = values.loc[values[values['AUC'] >= min_auc].index]
    # nbr_features = len(values.index)

    for mod in values.index:
        values.loc[mod,'pval'] = mannwhitneyu(X.iloc[y[y==1].index][mod],X.iloc[y[y==0].index][mod], alternative='two-sided')[1]

    values['qval'] = abs(multi.multipletests(values['pval'], method = 'fdr_bh')[1])
    values['pval'] = round(-np.log(values['pval'].astype(float))/np.log(10),3)
    values['qval'] = round(-np.log(values['qval'].astype(float))/np.log(10),3)

    values = values.sort_values(by=[sort], ascending=False)

    features = values.index.str.replace(' ',',').values.astype(str)
    features = [val[-1] for val in np.char.split(features,'+')]

    # Stacked circular bar plot

    theta = (2*np.pi)/nbr_features
    thetas = np.linspace(2*np.pi,0,nbr_features, endpoint=False)
    barWidth = theta
    barHeight = 1.0
    offset = 3.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    color_bar = list(Color('green',luminance=0.6).range_to(Color("red",luminance=0.5),nbr_features))
    for idx in range(len(color_bar)): color_bar[idx]=color_bar[idx].hex
    color_bar = pd.Series(color_bar)

    colors=pd.DataFrame(index=values.index,columns=values.columns)
    Max = values.max(axis=0)
    Min = values.min(axis=0)
    for mod in values.index:
        colors.loc[mod] = values.loc[mod].values
        colors.loc[mod] = ((colors.loc[mod]-Min)/(Max-Min) * nbr_features).astype(int)
        colors.loc[mod].loc[colors.loc[mod] == nbr_features] = nbr_features - 1
        colors.loc[mod] = color_bar[colors.loc[mod]].values

    ax.bar(thetas, barHeight, bottom=offset, color=colors['qval'], edgecolor='white', width=barWidth)
    ax.bar(thetas, barHeight, bottom=offset+barHeight, color=colors['pval'], edgecolor='white', width=barWidth) #bottom=values['qval']
    ax.bar(thetas, barHeight, bottom=offset+barHeight*2, color=colors['AUC'], edgecolor='white', width=barWidth) #bottom=np.add(values['qval'],values['pval']).tolist()

    # Legend and axis

    for val in list(values.columns):
        for idx in range(nbr_features):
            rot = np.rad2deg(thetas[idx])
            if idx in range(round(nbr_features/4),round(3*nbr_features/4)):
                rot = rot + 180
            ax.text(thetas[idx], offset+barHeight*list(values.columns)[::-1].index(val)+barHeight/2, values[val][idx],
                    fontsize=12, ha='center', va='center', rotation=rot)

    ax.set_xticks(thetas)
    ax.set_xticklabels(features)
    rotations = thetas
    rotations[np.cos(rotations) < 0] = rotations[np.cos(rotations) < 0] + np.pi
    rotations = np.rad2deg(rotations)
    labels = []
    for label, angle in zip(ax.get_xticklabels(), rotations):
        x,y = label.get_position()
        if x<=np.pi/2 or x>3*np.pi/2 : ha = 'left'
        else : ha = 'right'
        if x<np.pi : va = 'bottom'
        else : va = 'top'
        lab = ax.text(x,y+offset+(1+2*len(values.columns))*barHeight/2, label.get_text(), 
                        fontsize=12, ha=ha, va=va, rotation=angle)
        labels.append(lab)
    ax.set_xticklabels([])
    # ax.set_rticks([])
    ax.set_rgrids(offset+barHeight*np.array(range(len(values.columns)))+barHeight/2, 
                    labels=values.columns.values[::-1], angle=np.rad2deg(thetas[-1])/2, 
                    fontsize=12, fontweight="bold", ha='center')
    ax.grid(False)
    fig.subplots_adjust(bottom=0.2, top=0.8, left=0.2, right=0.8)
    
    # Show and save graphic
    plt.gcf().set_size_inches(15, 15)
    plt.savefig(out)
    # plt.show()

    print('Saving: ',os.path.basename(out))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input csv file (original or interraction features')
    parser.add_argument('--output','-o',default='out/circ.pdf',help='output filename')
    parser.add_argument('--sort',default='AUC',help='method for sorting values (AUC,pval,qval)')
    parser.add_argument('--original_features',default=0,help='number of original features without interractions')
    parser.add_argument('--nbr_features',type=int,help='number of features to show in plot')
    # parser.add_argument('--min_auc',default=0,help='minimum AUC to select features')
    args = parser.parse_args()

    main(args)
