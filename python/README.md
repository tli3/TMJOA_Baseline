# OAI

Contributors : Celia Le

Scripts for TMJOAI project

## Prerequisites

python 3.7.9 with the libraries : numpy pandas sklearn colour seaborn matplotlib statsmodels xgboost lightgbm

## What is it?

TMJOAI is a prediction tool of the health status of a patient for TemporoMandibular Joint Osteoarthitis (TMJ OA).

It uses decision tree algorithms for machine learning.

It has 2 features : 
* Making a prediction of the health status if the input file does not contain it : the output is a csv file containing the health prediction.
* Adding data to the dataset and train the machine learning models : the output contains the evaluation metrics of the model and several statistical plots (circplot, manhattanplot, Boxplot, ROC curve).

inputfile: csv file containing patient's data to predict health status -> Prediction | csv file containing data to add to the dataset -> Training 

File containing training dataset: Data.csv

## Running the code

run docker container tmjoai : bash src/main_TMJOAI.sh -i inputfile

The tool to use (prediction or training) is determined wheter the inputfile contains the health status of the patient or not.

run prediction tool : python3 src/main_prediction.py input --folder --output

run training tool : bash src/main_training.sh --input --datafile --output_folder --src_folder --model_folder --seed1 --seed_end --nbr_folds 

### Prediction

python3 src/main_prediction.py

input: csv file not containing the health status of a patient

output: csv file containing the prediction (healthy or diseased)

The prediction is based on the average prediction of the trained models (50 XGBoost models and 50 LightGBM models).

```
usage: main_prediction.py [-h] [--folder FOLDER] [--output OUTPUT] input

positional arguments:
  input                             input csv file with data to predict

optional arguments:
  -h, --help                        show this help message and exit
  --folder FOLDER, -f FOLDER        folder containing the models
  --output OUTPUT, -o OUTPUT        output file
```

### Training

bash src/main_training.sh

Input: csv file containg data to add to the training dataset

Output: trained models, evaluation metrics, statistical plots

What it does:
* Add data to Data.csv (containing the full dataset)
* Preprocess the data: Interaction file, AUC file
* Create statistical plots: circplot, manhattanplot (with and without interaction terms)
* Train the machine learning models: 5 models have been tested (XGBoost, LightGBM, RandomForest, RidgeRegression, LogisticRegression); 2 models are used to make the final prediction
* Calculate evaluation metrics: metrics of the different models trained, average of these metrics and metrics of the final model
* Create plots based on the models training: ROC, Boxplot_contribution, Boxplot_values

The models are trained using a default 10 times 5-folds cross-validation.

Each time, the random seed for spliting the folds for the cross validation varies between seed1 and seed_end.

```
Program to train the OA prediction tool

Syntax: main_training.sh [-i|d|o|s|m|seed1|seed_end|nbr_folds|h]
options:
-i|--inputfile         Name of the file containing the values to add to the training dataset.
-d|--datafile          Name of the file contraining all the training data.
-o|--output_folder     Name of the output folder to save the outputs.
-s|--src_folder        Name of the source folder containing the python scripts.
-m|--model_folder      Name of the source folder to save the trained models.
--seed1                First random seed to split the folds for the cross validation.
--seed_end             Last random seed to split the folds for the cross validation.
--nbr_folds            Number of folds for the cross validation.
-h|--help              Print this Help.
```

#### Add data to the dataset

python3 src/Step0_AddTrainingData.py

Verifies if the data in the inputfile is already in the file containing the dataset, if not, it adds the data at the end of the file.

```
usage: Step0_AddTrainingData.py [-h] [--file FILE] input

positional arguments:
  input                 input csv data file to add to the training dataset

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  csv file containing all the training data
```

#### Preprocess

python3 src/Step0_InterractionFile.py

Calculates the interaction beteween the features by multiplying them 2 by 2.

```
usage: Step0_InterractionFile.py [-h] [--output OUTPUT] input

positional arguments:
  input                         input csv file

optional arguments:
  -h, --help                    show this help message and exit
  --output OUTPUT, -o OUTPUT    output file
```

python3 src/Step0_AUC.py

Calculates the AUC of each feature.

```
usage: Step0_AUC.py [-h] [--input INPUT] [--output OUTPUT]
                    [--first_seed FIRST_SEED] [--last_seed LAST_SEED]
                    [--folds FOLDS]

optional arguments:
  -h, --help                    show this help message and exit
  --input INPUT, -i INPUT       input csv interraction file
  --output OUTPUT, -o OUTPUT    output filename
  --first_seed FIRST_SEED       number of the first seed
  --last_seed LAST_SEED         number of the last seed
  --folds FOLDS                 number of the folds for cross-validation
```

####  Create statistical plots

python3 src/STAT_circ.py

Draws circular plot containing the AUC, the pvalues and the qvalues of the features.

```
usage: STAT_circ.py [-h] [--output OUTPUT] [--sort SORT]
                    [--original_features ORIGINAL_FEATURES]
                    [--min_auc MIN_AUC]
                    input

positional arguments:
  input                 input csv file (original or interraction features

optional arguments:
  -h, --help                                show this help message and exit
  --output OUTPUT, -o OUTPUT                output filename
  --sort SORT                               method for sorting values (AUC,pval,qval)
  --original_features ORIGINAL_FEATURES     number of original features without interractions
  --min_auc MIN_AUC                         minimum AUC to select features
```

python3 src/STAT_manhattan.py

Draws manhattan plot of the AUC, pvalues and qvalues of the features.

```
usage: STAT_manhattan.py [-h] [--output OUTPUT]
                         [--original_features ORIGINAL_FEATURES]
                         input

positional arguments:
  input                                     input csv file (original or interraction features)

optional arguments:
  -h, --help                                show this help message and exit
  --output OUTPUT, -o OUTPUT                output filename
  --original_features ORIGINAL_FEATURES     number of original features to remove from interractions
```

#### Train the machine learning models

python3 src/Step1_RandomForest.py 

python3 src/Step1_RidgeRegression.py

python3 src/Step1_LogisticRegression.py

python3 src/Step1_XGBoost.py

```
usage: Step1_XGBoost.py [-h] [--interactions INTERACTIONS] [--auc AUC]
                        [--output OUTPUT]

optional arguments:
  -h, --help                                        show this help message and exit
  --interactions INTERACTIONS, -i INTERACTIONS      input csv interraction file
  --auc AUC                                         input csv AUC file
  --output OUTPUT, -o OUTPUT                        output folder
```

python3 src/Step1_LightGBM.py

```
usage: Step1_LightGBM.py [-h] [--interactions INTERACTIONS] [--auc AUC]
                        [--output OUTPUT]

optional arguments:
  -h, --help                                        show this help message and exit
  --interactions INTERACTIONS, -i INTERACTIONS      input csv interraction file
  --auc AUC                                         input csv AUC file
  --output OUTPUT, -o OUTPUT                        output folder
```

python3 src/Step1_FinalModel.py

Makes the average prediction of all the prediction made by the previously trained models.

```
usage: Step1_FinalModel.py [-h] [--interactions INTERACTIONS] [--auc AUC]
                           [--output OUTPUT] [--folder FOLDER]

optional arguments:
  -h, --help                                        show this help message and exit
  --interactions INTERACTIONS, -i INTERACTIONS      input csv interraction file to test
  --auc AUC                                         input csv AUC file
  --output OUTPUT, -o OUTPUT                        output folder
  --folder FOLDER, -f FOLDER                        models folder
```

#### Create plots based on the models training

python3 src/FinalStat.py

Returns the evaluation metrics of the trained models, their average and the metrics of the finale model.

```
usage: Step2_FinalStat.py [-h] [--output OUTPUT] [--folder FOLDER]

optional arguments:
  -h, --help                        show this help message and exit
  --output OUTPUT, -o OUTPUT        output filename
  --folder FOLDER                   folder to evaluate
```

#### Calculate evaluation metrics: metrics of the different models trained, average of these metrics and metrics of the final model

python3 src/Step2_ROC_Plot.py

Draws the ROC curve of the trained models and the top features.

```
usage: Step2_ROC_Plot.py [-h] [--input INPUT] [--output OUTPUT]
                         [--folder FOLDER]

optional arguments:
  -h, --help                        show this help message and exit
  --input INPUT, -i INPUT           input interaction features csv file
  --output OUTPUT, -o OUTPUT        output filename
  --folder FOLDER                   folder to evaluate
```

python3 src/Step2_Boxplot.py

Draws boxplot of the top features values and contributions.

```
usage: Step2_Boxplot.py [-h] [--input INPUT] [--output OUTPUT]
                        [--folder FOLDER]

optional arguments:
  -h, --help                        show this help message and exit
  --input INPUT, -i INPUT           input interraction features csv file
  --output OUTPUT, -o OUTPUT        output folder
  --folder FOLDER, -f FOLDER        folder to evaluate
```

