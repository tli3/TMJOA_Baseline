import argparse
import csv
import os

import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold

#########################################
#           Python 3.7.9                #
#           input = csv file            #
#       output = 'interractions.csv'    #
#########################################


def main(args):

    print('Adding data to the training dataset...')
    
    input = args.input
    datafile = args.file

    ###### Read file #####

    input_file = pd.read_csv(input, header=0)
    data = pd.read_csv(datafile, header=0)

    if ((data.sort_index(axis=1).columns != input_file.sort_index(axis=1).columns).any()):
        print('Invalid input file')
    else:
        for index in range(len(input_file)):
            if not ((data.sort_index(axis=1).values == input_file.sort_index(axis=1).iloc[index].values).all(axis=1).any()):
                data = data.append(input_file.iloc[index])
                print('Data added')
            else:
                print('Data '+str(index+1)+' already in the dataset')

        data.to_csv(datafile,index=False)
        print('Saving: ',os.path.basename(datafile))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input',help='input csv data file to add to the training dataset')
    parser.add_argument('--file','-f',default='Data.csv',help='csv file containing all the training data')
    args = parser.parse_args()

    main(args)
