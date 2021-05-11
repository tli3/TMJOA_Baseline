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
        Command="bash /app/OAI/python/src/main_training.sh -i /app/OAI/python/$(basename ${inputfile}) -d /app/OAI/python/Data.csv -o /app/OAI/python/output -s /app/OAI/python/src --seed1 1000 --seed_end 1010" #training
    else 
        Command="python3 /app/OAI/python/src/main_prediction.py /app/OAI/python/$(basename ${inputfile}) --folder /app/OAI/python/Models -o /app/OAI/python/output/Prediction.csv" #prediction
fi

# docker run --rm \
#     -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/Data.csv:/app/OAI/python/Data.csv \
#     -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/Models:/app/OAI/python/Models \
#     -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/${inputfile}:/app/OAI/python/$(basename ${inputfile}) \
#     -v /Users/luciacev-admin/Documents/TMJOAI/OAI_github/python/out:/app/OAI/python/output \
#     tmjoai:latest \
#     ${Command}

docker run --rm \
    -v /shiny-tooth/data/dcbia-filebrowser/source/tmjoai/Data.csv:/app/OAI/python/Data.csv \
    -v /shiny-tooth/data/dcbia-filebrowser/source/tmjoai/Models:/app/OAI/python/Models \
    -v /shiny-tooth/data/dcbia-filebrowser/${inputfile}:/app/OAI/python/$(basename ${inputfile}) \
    -v /shiny-tooth/data/dcbia-filebrowser/$(dirname ${inputfile}):/app/OAI/python/output \
    tmjoai:latest \
    ${Command}
