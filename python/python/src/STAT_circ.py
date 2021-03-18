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
    min_auc = float(args.min_auc)
    

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    modalities = input_file.columns.drop('y')[original_features:]
    X = input_file.loc[:,modalities] #value of covariates

    values = pd.DataFrame(index=modalities,columns=['AUC','pval','qval'])

    # Calculate AUC, pval and qval

    for mod in modalities:
        val = metrics.roc_auc_score(y, X[mod])
        values.loc[mod,'AUC'] = round(max(val,1-val),3)

    values = values.loc[values[values['AUC'] >= min_auc].index]
    nbr_features = len(values.index)

    for mod in values.index:
        values.loc[mod,'pval'] = wilcoxon(X.iloc[y[y==1].index][mod],X.iloc[y[y==0].index][mod], alternative='two-sided', correction=True)[1]
        # print(mod, values.loc[mod,'pval'])

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
    offset = 2.0

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

    bar_qval = ax.bar(thetas, barHeight, bottom=offset, color=colors['qval'], edgecolor='white', width=barWidth)
    bar_pval = ax.bar(thetas, barHeight, bottom=offset+barHeight, color=colors['pval'], edgecolor='white', width=barWidth) #bottom=values['qval']
    bar_AUC = ax.bar(thetas, barHeight, bottom=offset+barHeight*2, color=colors['AUC'], edgecolor='white', width=barWidth) #bottom=np.add(values['qval'],values['pval']).tolist()

    # Legend and axis

    for val in list(values.columns):
        for idx in range(nbr_features):
            rot = np.rad2deg(thetas[idx])
            if idx in range(round(nbr_features/4),round(3*nbr_features/4)):
                rot = rot + 180
            ax.text(thetas[idx], offset+barHeight*list(values.columns)[::-1].index(val)+barHeight/2, values[val][idx],
                    ha='center', va='center', rotation=rot)

    ax.set_xticks(thetas)
    ax.set_xticklabels(features)
    rotations = thetas
    rotations[np.cos(rotations) < 0] = rotations[np.cos(rotations) < 0] + np.pi
    rotations = np.rad2deg(rotations)
    labels = []
    for label, angle in zip(ax.get_xticklabels(), rotations):
        x,y = label.get_position()
        lab = ax.text(x,y+offset*1.2+len(values.columns)*barHeight+len(label.get_text())*0.04, label.get_text(), 
                        ha=label.get_ha(), va=label.get_va(), rotation=angle)
        labels.append(lab)
    ax.set_xticklabels([])
    # ax.set_rticks([])
    ax.set_rgrids(offset+barHeight*np.array(range(len(values.columns)))+barHeight/2, 
                    labels=values.columns.values[::-1], angle=np.rad2deg(thetas[-1])/2, 
                    fontsize=12, fontweight="bold", ha='center')
    ax.grid(False)
    
    # Show and save graphic
    plt.gcf().set_size_inches(15, 15)
    plt.savefig(out, format=out.split('.')[-1])
    # plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input csv file (original or interraction features')
    parser.add_argument('--output','-o',default='out/circ.pdf',help='output filename')
    parser.add_argument('--sort',default='AUC',help='method for sorting values (AUC,pval,qval)')
    parser.add_argument('--original_features',default=0,help='number of original features without interractions')
    parser.add_argument('--min_auc',default=0,help='minimum AUC to select features')
    args = parser.parse_args()

    main(args)
