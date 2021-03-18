import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pylab as plt

#########################################
#           Python 3.7.9                #
# input = 'interraction.csv', 'AUC.csv' #
#   output = 'Pred.csv', 'Stat.csv',    #
#   'Importance.csv', 'Importance.txt'  #
#########################################

def main(args):

    interractions = args.interractions
    auc = args.auc
    out = args.output
    seed1 = int(args.first_seed)
    seed_end = int(args.last_seed)
    nbr_seed = seed_end-seed1
    nbr_folds = int(args.folds)

    if out[-1]!='/':
        out=out+'/'

    if not os.path.exists(out):
        os.mkdir(out)

    interractions = pd.read_csv(interractions)
    AUC = pd.read_csv(auc, index_col=0)

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
            index = AUC[AUC[str(seed)+'_'+str(i+1)] > 0.68].index
            X_train, X_test = X.loc[train_index][index], X.loc[test_index][index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # Grid Search
            # gsc = GridSearchCV(
            #     estimator=RandomForestRegressor(random_state=0),
            #     param_grid={
            #         'n_estimators': (50, 100, 200, 500, 1000, 2000),
            #     },
            #     cv=5, verbose=0,n_jobs=-1)
            
            # grid_result = gsc.fit(X_train, y_train)
            # best_params = grid_result.best_params_
            # print(best_params)
            # rfr = RandomForestRegressor(n_estimators=best_params["n_estimators"],random_state=0, verbose=False)

            rfr = RandomForestRegressor(n_estimators=500,random_state=0, verbose=False)
            rfr.fit(X_train, y_train)
            pred.at[test_index,seed] = rfr.predict(X_test).astype(float)
            importance.at[index,str(seed)+'_'+str(i+1)] = rfr.feature_importances_
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

    mean = importance.mean(axis=1)
    importancetxt = pd.DataFrame(mean, index = mean.index, columns = ['mean'])
    importancetxt = importancetxt.sort_values(by=['mean'], ascending=False)

    pred.to_csv(out+'Pred.csv', index=False)
    importance.to_csv(out+'Importance.csv')
    stat.to_csv(out+'Stat.csv')
    importancetxt.to_csv(out+'Importance.txt', header=False)

    df = pd.DataFrame(importancetxt[(importancetxt>0.01).any(1)], columns=['mean'])
    df['mean'] = df['mean'].fillna(0).astype(float)

    plt.figure()
    df.plot()
    df.plot(kind='bar', y='mean', legend=False, figsize=(30, 30))
    plt.title('XGBoost Feature Importance')
    plt.ylabel('relative importance')
    plt.gcf().savefig(out+'feature_importance.png')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--interractions','-i',default='interractions.csv',help='input csv interraction file')
    parser.add_argument('--auc',default='AUC.csv',help='input csv AUC file')
    parser.add_argument('--output','-o',default='RandomForest/',help='output folder')
    parser.add_argument('--first_seed',default=2020,help='number of the first seed')
    parser.add_argument('--last_seed',default=2030,help='number of the last seed')
    parser.add_argument('--folds',default=5,help='number of the folds for cross-validation')
    args = parser.parse_args()

    main(args)
