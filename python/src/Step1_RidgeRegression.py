import os
import pickle 
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pylab as plt
from sklearn import preprocessing

#########################################
#           Python 3.7.9                #
# input = 'interraction.csv', 'AUC.csv' #
#   output = 'Pred.csv', 'Stat.csv'     #
#########################################

def main(args):

    print('Training RidgeRegression model')

    interractions = args.interractions
    auc = args.auc
    out = args.output

    if out[-1]!='/':
        out=out+'/'

    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

    interractions = pd.read_csv(interractions)
    AUC = pd.read_csv(auc, index_col=0)
    seed1 = int(AUC.columns[0].split('_')[0])
    seed_end = int(AUC.columns[-1].split('_')[0]) + 1
    nbr_seed = seed_end-seed1
    nbr_folds = int(AUC.columns[-1].split('_')[-1])

    modalities = interractions.columns.drop('y')
    y = interractions['y'] #result(0 or 1)
    X = interractions.loc[:,modalities] #value of covariates
    nbr_features = len(modalities)
    samples = len(y)

    stat = pd.DataFrame(columns=['ACC','PREC1','PREC0','RECALL1','RECALL0','F1SCORE','AUC'])
    importance = pd.DataFrame(0, index=modalities, columns=AUC.columns)
    pred = pd.DataFrame(index=range(samples),columns=range(seed1,seed_end))

    for seed in range(seed1,seed_end):
        print(seed)
        skf = StratifiedKFold(n_splits=nbr_folds, shuffle = True, random_state = seed)
        skf.get_n_splits(X, y)

        i=0
        for train_index, test_index in skf.split(X, y):
            index = AUC[AUC[str(seed)+'_'+str(i+1)] > 0.7].index
            X_train, X_test = X.loc[train_index][index], X.loc[test_index][index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            clf = RidgeCV(alphas=[1, 10, 25, 50, 75, 100, 200], cv=5)
            clf = clf.fit(X_train, y_train)
            best_alpha = clf.alpha_
            # print(best_alpha)

            clf = Ridge(alpha=best_alpha, normalize=True, random_state=0)
            clf = clf.fit(X_train, y_train)
            pred.at[test_index,seed] = clf.predict(X_test)
            pickle.dump(clf, open(out+'RidgeRegression_'+str(seed)+'_'+str(i+1)+'.pkl', 'wb'))


            i+=1

        pred_seed = pred[seed].astype(float).round(0).astype(bool)
        y_bool = y.astype(bool)
        acc = round(metrics.accuracy_score(y_bool, pred_seed), 4)
        prec1 = round(metrics.precision_score(y_bool, pred_seed), 4)
        prec0 = round(metrics.precision_score(~y_bool, ~pred_seed), 4)
        recall1 = round(metrics.recall_score(y_bool, pred_seed), 4)
        recall0 = round(metrics.recall_score(~y_bool, ~pred_seed), 4)
        f1 = round(metrics.f1_score(y_bool, pred_seed), 4)
        auc = round(metrics.roc_auc_score(y_bool, pred[seed]), 4)
        stat.loc[seed-seed1] = [acc,prec1,prec0,recall1,recall0,f1,auc]

    pred.to_csv(out+'Pred.csv', index=False)
    stat.to_csv(out+'Stat.csv')

    print('Model saved')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--interractions','-i',default='interractions.csv',help='input csv interraction file')
    parser.add_argument('--auc',default='AUC.csv',help='input csv AUC file')
    parser.add_argument('--output','-o',default='RidgeRegression/',help='output folder')
    args = parser.parse_args()

    main(args)
