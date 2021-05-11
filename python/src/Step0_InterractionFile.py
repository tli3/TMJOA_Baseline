import argparse
import csv
import os
import sys

import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#       output = 'interactions.csv'    #
#########################################


def main(args):
    
    input = args.input
    out = args.output

    print('Creating: ',os.path.basename(out))
    
    ##### Creating folders #####

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass

    ###### Read file #####

    input_file = pd.read_csv(input)
    X = input_file.drop(['y'], axis=1, errors='ignore')
    modalities = X.columns
    nbr_features = len(modalities)

    ##### Interaction features file #####

    features = input_file
    for m1 in range(nbr_features):
        for m2 in range(m1,nbr_features):
            if m1 != m2:

                split1 = modalities[m1].split('+')
                split2 = modalities[m2].split('+')
                feature_name = split1[0]+'*'+split2[0]+'+'+split1[1]+'*'+split2[1]
                new_feature = X[modalities[m1]]*X[modalities[m2]]
                features[feature_name] = new_feature

    features.to_csv(out,index=False)
    print('Saving: ',os.path.basename(out))
    return features

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input csv file')
    parser.add_argument('--output','-o',default='interactions.csv',help='output file')
    args = parser.parse_args()

    main(args)
