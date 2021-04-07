#!/bin/sh

while getopts "i:" flag
do
    case "${flag}" in
        i) inputfile=${OPTARG};;
    esac
done

echo ${inputfile}

IFS=',' read -a var < ${inputfile}

if [[ " ${var[@]} " =~ " y " ]]
    then
        Command="bash /app/src/main_training.sh -i ${inputfile} -o /app/out -s /app/src --seed1 1000 --seed_end 1010" #training
    else 
        Command="python3 /app/src/main_prediction.py ${inputfile} --folder /app/Models -o /app/out/Prediction.csv" #prediction
fi

docker run --rm \
    -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/Data.csv:/app/Data.csv \
    -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/Models:/app/Models \
    -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/${inputfile}:/app/$(basename ${inputfile}) \
    -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/out:/app/out \
    tmjoai:latest \
    ${Command}
    # -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/src:/app/src \


# docker run --rm \
#     -v /shiny-tooth/data/dcbia-filebrowser/source/tmjoai/Data.csv:/app/Data.csv \
#     -v /shiny-tooth/data/dcbia-filebrowser/source/tmjoai/Models:/app/Models \
#     -v /shiny-tooth/data/dcbia-filebrowser/source/tmjoai/src:/app/src \
#     -v /shiny-tooth/data/dcbia-filebrowser/${inputfile}:/app/$(basename ${inputfile}) \
#     -v /shiny-tooth/data/dcbia-filebrowser/$(dirname ${inputfile}):/app/out \
#     tmjoai:latest \
#     ${Command}
