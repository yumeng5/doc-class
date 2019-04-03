#!/bin/bash

# dataset directory
dataset=agnews

# text file name; one document per line
text_file=text.txt

# weak supervision source
sup_source=names

# whether ground truth labels are available for evaluation; True or False
with_evaluation=False

# use GPU 
export CUDA_VISIBLE_DEVICES=7

green=`tput setaf 2`
reset=`tput sgr0`

if [ ! -e ./${dataset}/cleaned.txt ] 
then
    echo ${green}===Step 1: Pre-processing===${reset}
    python preprocess.py --dataset ${dataset} --in_file ${text_file} --out_file cleaned.txt
else
    echo ${green}===Step 1: Pre-processing Skipped\; Using Pre-processed File===${reset}
fi    

if [ ! -e ./${dataset}/embedding.txt ] 
then
    echo ${green}===Step 2: Word Embedding Training===${reset}
    ./word2vec -train ./${dataset}/cleaned.txt -output ./${dataset}/embedding.txt -kappa ./${dataset}/sep.txt -topic ./${dataset}/classes.txt -reg_lambda 1 -cbow 0 -size 100 -global_lambda 1.5 -window 5 -negative 5 -sample 1e-3 -min-count 5 -threads 20 -binary 0 -iter 10
else
    echo ${green}===Step 2: Word Embedding Training Skipped\; Using Trained Embedding===${reset}
fi

echo ${green}===Step 3: Classification===${reset}

python main.py --dataset ${dataset} --sup_source ${sup_source} --with_evaluation ${with_evaluation}
