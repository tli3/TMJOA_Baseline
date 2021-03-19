#!/bin/sh

Help()
{
   # Display Help
   echo "Program to train the OA prediction tool"
   echo
   echo "Syntax: main_training.sh [-i|d|o|s|m|seed1|seed_end|nbr_folds|h]"
   echo "options:"
   echo "-i              Name of the file containing the values to add to the training dataset."
   echo "-d              Name of the file contraining all the training data."
   echo "-o              Name of the output folder to save the outputs."
   echo "-s              Name of the source folder containing the python scripts."
   echo "-m              Name of the source folder to save the trained models."
   echo "--seed1          First random seed to split the folds for the cross validation."
   echo "--seed_end       Last random seed to split the folds for the cross validation."
   echo "--nbr_folds      Number of folds for the cross validation."
   echo "-h              Print this Help."
   echo
}

# inputfile=$1
# shift

# if [ -z $inputfile ];
# then
#     echo " - Missing mandatory argument: inputfile. "
#     Help
#     exit 1
# fi

while [ "$1" != "" ]; do
    case $1 in
        -i | --inputfile )  shift
            inputfile=$1;;
        -d | --datafile)  shift
            inputfile=$1;;
        -o | --output_folder )  shift
            output_folder=$1;;
        -s | --src_folder ) shift
            src_folder=$1;;
        -m | --model_folder ) shift
            model_folder=$1;;
        --seed1 )   shift
            seed1=$1;;
        --seed_end )    shift
            seed_end=$1;;
        --nbr_folds )   shift
            nbr_folds=$1;;
        -h | --help )
            Help
            exit;;
        * ) 
            echo ' - Error: Unsupported flag'
            Help
            exit 1
    esac
    shift
done

Data="${datafile:-Data.csv}"
output_folder="${output_folder:-out}"
src_folder="${src_folder:-src}"
model_folder="${model_folder:-Models}"
seed1="${seed1:=2020}"
seed_end="${seed_end:=2030}"
nbr_folds="${nbr_folds:=5}"

nbr_features=$((`head -1 Data.csv | sed 's/[^,]//g' | wc -m` -1))

# echo ${inputfile}

python3 ${src_folder}/Step0_AddTrainingData.py ${inputfile} --file ${Data}
python3 ${src_folder}/Step0_InterractionFile.py ${Data} -o interactions.csv
python3 ${src_folder}/Step0_AUC.py -i interactions.csv -o AUC.csv --first_seed ${seed1} --last_seed ${seed_end} --folds ${nbr_folds}
python3 ${src_folder}/STAT_circ.py ${Data} -o ${output_folder}/circ.pdf --sort pval
python3 ${src_folder}/STAT_circ.py interactions.csv -o ${output_folder}/circ_interactions.pdf --sort AUC --original_features ${nbr_features} --min_auc 0.65
python3 ${src_folder}/STAT_manhattan.py ${Data} -o ${output_folder}/manhattan.pdf
python3 ${src_folder}/STAT_manhattan.py interactions.csv -o ${output_folder}/manhattan_interactions.pdf --original_features ${nbr_features}

for modelName in XGBoost LightGBM 
do
    python3 ${src_folder}/Step1_${modelName}.py --interactions interactions.csv --auc AUC.csv -o ${model_folder}/${modelName}
    # python3 ${src_folder}/Step2_Boxplot.py -i interactions.csv -o ${output_folder} --folder ${model_folder}${modelName}
    # python3 ${src_folder}/Step2_ROC_Plot.py -i interactions.csv -o ${output_folder}/ROC.pdf --folder ${model_folder}${modelName}
done

python3 ${src_folder}/Step1_FinalModel.py --interactions interactions.csv --auc AUC.csv -o ${model_folder}/FinalModel --folder ${model_folder}
python3 ${src_folder}/Step2_Boxplot.py -i interactions.csv -o ${output_folder} --folder ${model_folder}/FinalModel
python3 ${src_folder}/Step2_ROC_Plot.py -i interactions.csv -o ${output_folder}/ROC.pdf --folder ${model_folder}

python3 ${src_folder}/Step2_FinalStat.py -o ${output_folder}/Stats.csv --folder ${model_folder}
