# OAI
Scripts for OAI project

Input: OAI_20190621.csv

# step1. preprocess

python3 Step0_InterractionFile.py input --output

python3 Step0_AUC.py --input --output

# step2. draw circplot, mahplot with and without interaction terms.

Rscript STAT_circ.r

Rscript STAT_mahplot.r

Rscript STAT_circ_Inter.r

Rscript STAT_mahplot_inter.r

# step3. Prediction

python3 Step1_RandomForest.py
python3 Step1_XGBoost.py
python3 Step1_LightGBM.py
python3 Step1_RidgeRegression.py
python3 Step1_LogisticRegression.py

Rscript Pred_step2_inference_plot.r
