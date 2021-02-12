import os
import csv
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

#########################################
#           Python 3.7.9                #
#       input = 'interraction.csv'      #
#           output = 'AUC.csv'          #
#########################################

def main(args):

    ##### Create folds CV + compute AUC #####
    input = args.input
    out = args.output

    if out[-1]!='/':
        out=out+'/'

    if not os.path.exists(out):
        os.mkdir(out)

    seed1 = int(args.first_seed)
    seed_end = int(args.last_seed)
    nbr_seed = seed_end-seed1
    nbr_folds = int(args.folds)

    interractions = pd.read_csv(input)
    y = interractions['y'] #result(0 or 1)
    modalities = interractions.columns.drop('y')
    X = interractions.loc[:,modalities] #value of covariates
    nbr_features = len(modalities)

    AUC = pd.DataFrame(index=list(modalities)) #, columns=list(col))

    for seed in range(seed1,seed_end):
        print(seed)
        skf = StratifiedKFold(n_splits=nbr_folds, shuffle = True, random_state = seed)
        skf.get_n_splits(X, y)
        i=0
        for train_index, test_index in skf.split(X, y):
            # # print(i)
            # print("TRAIN:", train_index, "TEST:", test_index)

            X_train = X.loc[train_index]
            X_test = X.loc[test_index]
            y_train = y.loc[train_index]
            y_test = y.loc[test_index]
            auc = []

            for mod in modalities:
                val = metrics.roc_auc_score(y_train, X_train[mod])
                auc.append(max(val,1-val)) # trick to get the same results as the R script
            AUC[str(seed)+'_'+str(i+1)] = auc
            i+=1
    AUC.to_csv(out+'AUC.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',default='interractions.csv',help='input csv interraction file')
    parser.add_argument('--output','-o',default='./',help='output folder')
    parser.add_argument('--first_seed',default=2020,help='number of the first seed')
    parser.add_argument('--last_seed',default=2030,help='number of the last seed')
    parser.add_argument('--folds',default=5,help='number of the folds for cross-validation')
    args = parser.parse_args()

    main(args)
