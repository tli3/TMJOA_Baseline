# OAI

Scripts for OAI project

Input: OAI_20190621.csv

# step1. preprocess

python3 Step0_InterractionFile.py input --output

python3 Step0_AUC.py --input --output

# step2. draw circplot, mahplot with and without interaction terms.

python3 STAT_circ.py

python3 STAT_manhattan.py

# step3. Prediction

python3 Step1_RandomForest.py

python3 Step1_XGBoost.py

python3 Step1_LightGBM.py

python3 Step1_RidgeRegression.py

python3 Step1_LogisticRegression.py



python3 Step2_ROC_Plot.py

python3 Step2_Boxplot.py


