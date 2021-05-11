import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

#########################################
#           Python 3.7.9                #
# input = 'interraction.csv', 'AUC.csv' #
#   output = 'Pred.csv', 'Stat.csv',    #
#   'Importance.csv', 'Importance.txt'  #
#########################################

def main(args):

    print('Training: XGBoost model')

    interactions = args.interactions
    auc = args.auc
    out = args.output

    if out[-1]!='/':
        out=out+'/'

    if not os.path.exists(os.path.dirname(out+'models/')):
        try:
            os.makedirs(os.path.dirname(out+'models/'))
        except:
            pass

    interactions = pd.read_csv(interactions)
    AUC = pd.read_csv(auc, index_col=0)
    seed1 = int(AUC.columns[0].split('_')[0])
    seed_end = int(AUC.columns[-1].split('_')[0]) + 1
    nbr_seed = seed_end-seed1
    nbr_folds = int(AUC.columns[-1].split('_')[-1])

    modalities = interactions.columns.drop('y')
    y = interactions['y'] #result(0 or 1)
    X = interactions.loc[:,modalities] #value of covariates
    nbr_features = len(modalities)
    samples = len(y)

    # PARA = pd.DataFrame([[0.01,1,0.5,0.5,5000],[0.01,1,0.7,0.5,5000],[0.001,1,0.7,0.5,10000],[0.01,2,0.5,0.5,5000],[0.001,2,0.5,0.5,10000],[0.01,2,0.7,0.5,5000],[0.001,2,0.7,0.5,10000]], columns=['eta','W','C','S','num_round'])
    PARA = pd.DataFrame([[0.01,2,0.7,0.5,5000]], columns=['eta','W','C','S','num_round'])

    for para in range(len(PARA)):
        print('Para',para+1,'/',len(PARA))
        eta=PARA.at[para,'eta']
        W=PARA.at[para,'W']
        C=PARA.at[para,'C']
        S=PARA.at[para,'S']
        # output = out+'eta'+str(eta)+'W'+str(W)+'C'+str(C)+'S'+str(S)+'/'
        # if not os.path.exists(output):
        #     os.mkdir(output)

        stat = pd.DataFrame(columns=['ACC','PREC1','PREC0','RECALL1','RECALL0','F1SCORE','AUC'])
        importance = pd.DataFrame(0, index=modalities, columns=AUC.columns)
        pred = pd.DataFrame(index=range(samples),columns=range(seed1,seed_end))
        features = pd.DataFrame()

        for seed in range(seed1,seed_end):
            print('Seed: ',str(seed))
            skf = StratifiedKFold(n_splits=nbr_folds, shuffle = True, random_state = seed)
            skf.get_n_splits(X, y)

            i=0
            for train_index, test_index in skf.split(X, y):
                model = str(seed)+'_'+str(i+1)
                AUC_seed = AUC[model]
                index = AUC_seed[AUC_seed > 0.68].index
                AUC_seed = AUC_seed.loc[index]
                X_train, X_test = X.loc[train_index][index], X.loc[test_index][index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]

                # Correlation-based features selection
                correlation = X_train.corr()
                correlation.to_numpy()[np.tril_indices(len(correlation))] = 0
                while (abs(correlation)).max().max() > 0.65:
                    row = abs(correlation).max(1).idxmax()
                    col = abs(correlation).max(0).idxmax()
                    if AUC_seed.loc[row]>= AUC_seed.loc[col]:
                        X_train = X_train.loc[:,X_train.columns.drop(col)]
                        AUC_seed = AUC_seed.loc[AUC_seed.index.drop(col)]
                    else:
                        X_train = X_train.loc[:,X_train.columns.drop(row)]
                        AUC_seed = AUC_seed.loc[AUC_seed.index.drop(row)]
                    correlation = X_train.corr()
                    correlation.to_numpy()[np.tril_indices(len(correlation))] = 0
                new_features = pd.DataFrame({model: X_train.columns})
                features = pd.concat([features,new_features], ignore_index=False, axis=1)
                X_test = X_test.loc[:,features[model].dropna()]

                num_round = PARA.at[para,'num_round']
                param = {'objective':'binary:logistic', 'n_estimators':num_round, 'subsample':S, 'max_depth':1, 'colsample_bytree':C, 'eta':eta, 'eval_metric':'auc', 'min_child_weight':W, 'scale_pos_weight':sum(np.logical_not(y_train))/sum(y_train)}

                # Nested CV
                skf0 = StratifiedKFold(n_splits=nbr_folds, shuffle = True, random_state = 0)
                skf0.get_n_splits(X_train, y_train)
                X_train.index = list(range(len(train_index)))
                y_train.index = list(range(len(train_index)))
                for train_ind, test_ind in skf0.split(X_train, y_train):
                    X_train_in, X_test_in = X_train.loc[train_ind], X_train.loc[test_ind]
                    y_train_in, y_test_in = y_train.loc[train_ind], y_train.loc[test_ind]
                    bst = xgb.XGBRegressor(**param)
                    bst.fit(X_train, y_train, eval_set = [(X_train_in,y_train_in), (X_test_in, y_test_in)],eval_metric = 'rmse',early_stopping_rounds = 100, verbose=False)
                param['n_estimators'] = bst.get_booster().best_iteration
                # print(param['n_estimators'])

                bst = xgb.XGBRegressor(**param).fit(X_train,y_train)
                pred.at[test_index,seed] = bst.predict(X_test)
                importance.at[features[model].dropna(),model] = bst.feature_importances_
                pickle.dump(bst, open(out+'models/XGBoost_'+model+'.pkl', 'wb'))

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
            stat.loc['XGBoost_'+str(seed)] = [acc,prec1,prec0,recall1,recall0,f1,auc]

        stat.loc['mean'] = stat.mean().round(4)
        print(stat.loc['mean'])

        mean = importance.mean(axis=1)
        importancetxt = pd.DataFrame(mean, index = mean.index, columns = ['mean'])
        importancetxt = importancetxt.sort_values(by=['mean'], ascending=False)
        pred.to_csv(out+'Predictions.csv')
        importance.to_csv(out+'Importance.csv')
        stat.to_csv(out+'Stat.csv')
        importancetxt.to_csv(out+'Importance.txt', header=False)
        features.to_csv(out+'Features.csv')

        df = pd.DataFrame(importancetxt[(importancetxt>0.01).any(1)], columns=['mean'])
        df['mean'] = df['mean'].fillna(0).astype(float)

        print('Saving: XGBoost model')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions','-i',default='interactions.csv',help='input csv interraction file')
    parser.add_argument('--auc',default='AUC.csv',help='input csv AUC file')
    parser.add_argument('--output','-o',default='Models/XGBoost/',help='output folder')
    args = parser.parse_args()

    main(args)

