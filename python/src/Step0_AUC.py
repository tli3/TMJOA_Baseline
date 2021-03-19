import argparse
import csv
import operator
import os

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

#########################################
#           Python 3.7.9                #
#       input = 'interraction.csv'      #
#           output = 'AUC.csv'          #
#########################################

def main(args):

    ##### Create folds CV + compute AUC #####
    input = args.input
    out = args.output

    print('Creating: ',os.path.basename(out))

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass

    seed1 = int(args.first_seed)
    seed_end = int(args.last_seed)
    nbr_seed = seed_end-seed1
    nbr_folds = int(args.folds)

    interactions = pd.read_csv(input)
    y = interactions['y'] #result(0 or 1)
    modalities = interactions.columns.drop('y')
    X = interactions.loc[:,modalities] #value of covariates
    # X = (X-X.mean())/X.std()
    nbr_features = len(modalities)

    AUC = pd.DataFrame(index=list(modalities)) #, columns=list(col))

    for seed in range(seed1,seed_end):
        print('Seed: ',str(seed))
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
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for label in [0,1]:
                    fpr[label], tpr[label], _ = metrics.roc_curve(y_train, X_train[mod], pos_label=label)
                    roc_auc[label] = round(metrics.auc(fpr[label], tpr[label]),3)
                ind_max = max(roc_auc.items(), key=operator.itemgetter(1))[0]
                auc.append(roc_auc[ind_max])
            AUC[str(seed)+'_'+str(i+1)] = auc
            i+=1
    AUC.to_csv(out)
    print('Saving: ',os.path.basename(out))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',default='interactions.csv',help='input csv interraction file')
    parser.add_argument('--output','-o',default='AUC.csv',help='output filename')
    parser.add_argument('--first_seed',default=2020,help='number of the first seed')
    parser.add_argument('--last_seed',default=2030,help='number of the last seed')
    parser.add_argument('--folds',default=5,help='number of the folds for cross-validation')
    args = parser.parse_args()

    main(args)
