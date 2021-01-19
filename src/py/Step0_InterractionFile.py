import os
import csv
import pandas as pd
import argparse
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#           output = 'AllT.csv'         #
#########################################


def main(args):
    
    input = args.input
    out = args.output
    
    ##### Creating folders #####

    if not os.path.exists(out):
        os.mkdir(out)

    for fold in ['out','RandomForest','XGBoost','LightGBM','Ridge','Logistic']:
        if not os.path.exists(out+fold):
            os.mkdir(out+fold)

    ###### Read file #####

    input_file = pd.read_csv(input)
    y = input_file['y'] #result(0 or 1)
    X = input_file.loc[:,input_file.columns != 'y'] #value of covariates

    modalities = input_file.columns.drop('y')
    nbr_features = len(modalities)

    # ##### Interaction features file #####

    features = input_file
    for m1 in range(nbr_features):
        for m2 in range(m1,nbr_features):
            if m1 != m2:

                split1 = modalities[m1].split('+')
                split2 = modalities[m2].split('+')
                feature_name = split1[0]+'*'+split2[0]+'+'+split1[1]+'*'+split2[1]
                new_feature = X[modalities[m1]]*X[modalities[m2]]
                features[feature_name] = new_feature

    features.to_csv(out+'interractions.csv',index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input',help='input csv file')
    parser.add_argument('--output','-o',default='./',help='output folder')
    args = parser.parse_args()

    main(args)
