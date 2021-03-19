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
#           input = models folder       #
#           output = 'Stats.csv'        #
#########################################

def main(args):

    print('Calculating statistical results')

    out = args.output
    folder = args.folder

    if folder[-1]!='/':
        folder=folder+'/'

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass

    stat_XGBoost = pd.read_csv(folder+'XGBoost/Stat.csv', index_col=0)
    stat_LightGBM = pd.read_csv(folder+'LightGBM/Stat.csv', index_col=0)
    Stats = stat_XGBoost.drop(['mean']).append(stat_LightGBM.drop(['mean']))

    Stats.loc['mean'] = Stats.mean().round(4)
    Stats.loc['FinalModel'] = pd.read_csv(folder+'FinalModel/Stat.csv').loc[0]
    Stats.to_csv(out)
    print('Saving: ',os.path.basename(out))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output','-o',default='out/Stats.csv',help='output filename')
    parser.add_argument('--folder',default='Models/',help='folder to evaluate')
    args = parser.parse_args()

    main(args)
