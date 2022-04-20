#!/bin/sh

SECONDS=0
Help()
{
   # Display Help
   echo "Program to train the OA prediction tool"
   echo
   echo "Syntax: main_training.sh [--OPTIONS]"
   echo "options:"
   echo "-i|--inputfile         Name of the file containing the values to add to the training dataset."
   echo "-d|--datafile          Name of the file contraining all the training data."
   echo "--interaction_file     Name of the file contraining the interactions features calculated from the training data."
   echo "-a|--auc               Name of the file contraining the AUC value of each interaction feature."
   echo "-o|--output_folder     Name of the output folder to save the outputs."
   echo "-s|--src_folder        Name of the source folder containing the python scripts."
   echo "-m|--model_folder      Name of the source folder to save the trained models."
   echo "--seed1                First random seed to split the folds for the cross validation."
   echo "--seed_end             Last random seed to split the folds for the cross validation."
   echo "--nbr_folds            Number of folds for the cross validation."
   echo "-h|--help              Print this Help."
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
            datafile=$1;;
        --interaction_file)  shift
            interaction_file=$1;;
        -a | --auc)  shift
            auc_file=$1;;
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

datafile="${datafile:-Data.csv}"
interaction_file="${interaction_file:-interactions.csv}"
auc_file="${auc_file:-AUC.csv}"
output_folder="${output_folder:-out}"
src_folder="${src_folder:-src}"
model_folder="${model_folder:-Models}"
seed1="${seed1:=1000}"
seed_end="${seed_end:=1010}"
nbr_folds="${nbr_folds:=5}"

nbr_features=$((`head -1 $datafile | sed 's/[^,]//g' | wc -m` -1))

echo ${auc_file}

if [ -z $inputfile ];
then
    echo " - Missing inputfile, will not append data"
else
    python3 ${src_folder}/Step0_AddTrainingData.py ${inputfile} --file ${datafile}
fi

python3 ${src_folder}/Step0_InterractionFile.py ${datafile} -o ${interaction_file}
python3 ${src_folder}/Step0_AUC.py -i ${interaction_file} -o $auc_file --first_seed ${seed1} --last_seed ${seed_end} --folds ${nbr_folds}
python3 ${src_folder}/STAT_circ.py ${datafile} -o ${output_folder}/circ.pdf --sort AUC
python3 ${src_folder}/STAT_circ.py ${interaction_file} -o ${output_folder}/circ_interactions.pdf --sort AUC --original_features ${nbr_features} #--min_auc 0.65
python3 ${src_folder}/STAT_manhattan.py ${datafile} -o ${output_folder}/manhattan.pdf
python3 ${src_folder}/STAT_manhattan.py ${interaction_file} -o ${output_folder}/manhattan_interactions.pdf --original_features ${nbr_features}

for modelName in XGBoost LightGBM 
do
    python3 ${src_folder}/Step1_${modelName}.py --interactions ${interaction_file} --auc $auc_file -o ${model_folder}/${modelName}
    # python3 ${src_folder}/Step2_Boxplot.py -i ${interaction_file} -o ${output_folder} --folder ${model_folder}${modelName}
    # python3 ${src_folder}/Step2_ROC_Plot.py -i ${interaction_file} -o ${output_folder}/ROC.pdf --folder ${model_folder}${modelName}
done

python3 ${src_folder}/Step1_FinalModel.py --interactions ${interaction_file} --auc $auc_file -o ${model_folder}/FinalModel --folder ${model_folder}
python3 ${src_folder}/Step2_Boxplot.py -i ${interaction_file} -o ${output_folder} --folder ${model_folder}/FinalModel
python3 ${src_folder}/Step2_ROC_Plot.py -i ${interaction_file} -o ${output_folder}/ROC.pdf --folder ${model_folder}

python3 ${src_folder}/Step2_FinalStat.py -o ${output_folder}/Stats.csv --folder ${model_folder}

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
