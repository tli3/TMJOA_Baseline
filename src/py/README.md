# OAI
Scripts for OAI project

Input: OAI_20190621.csv

# step1. preprocess

python3 Step0_InterractionFile.py --input --output

python3 Step0_AUC.py --input --output

# step2. draw circplot, mahplot with and without interaction terms.

Rscript STAT_circ.r

Rscript STAT_mahplot.r

Rscript STAT_circ_Inter.r

Rscript STAT_mahplot_inter.r

# step3. Prediction

Rscript Pred_step1_data_prediction.r

Rscript Pred_step2_inference_plot.r
