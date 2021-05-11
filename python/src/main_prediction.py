import argparse
import csv
import os
import pickle
from argparse import Namespace

import numpy as np
import pandas as pd
from sklearn import metrics

import Step0_InterractionFile

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#       output = 'Predictions.csv'      #
#########################################


def main(args):
    
    print('Prediction in progress...')

    input = args.input
    out = args.output
    folder = args.folder

    if folder[-1]!='/':
        folder=folder+'/'

    if not os.path.exists(os.path.dirname(out)):
        try:
            os.makedirs(os.path.dirname(out))
        except:
            pass
    
    interactions = Step0_InterractionFile.main(Namespace(input=input, output='interationsInput.csv'))
    # interactions = pd.read_csv('interationsInput.csv')
    
    models = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.pkl'):
                models.append(os.path.join(root,file))

    nbr_models = len(models)
    pred = np.zeros(len(interactions))

    for modelname in models:
        model = pickle.load(open(modelname, 'rb'))
        splitname = modelname.split('.')[0].split('_')
        seed = splitname[1]
        fold = splitname[2]
        features = pd.read_csv(os.path.dirname(modelname)+'/../Features.csv')[seed+'_'+fold].dropna()
        fileToPred = interactions.loc[:,features]
        pred = np.add(pred,np.array(model.predict(fileToPred)))
    

    print(pred/nbr_models)
    pred = [np.round(x/nbr_models) for x in pred]
    Predictions = pd.DataFrame({'Pred':pred}, index=interactions.index)
    Predictions.loc[Predictions['Pred']==1] = 'Diseased'
    Predictions.loc[Predictions['Pred']==0] = 'Healthy'
    Predictions.to_csv(out)

    print('Saving: ',os.path.basename(out))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input csv file with data to predict')
    parser.add_argument('--folder','-f',default='Models/',help='folder containing the models')
    parser.add_argument('--output','-o',default='out/Prediction.csv',help='output file')
    args = parser.parse_args()

    main(args)
