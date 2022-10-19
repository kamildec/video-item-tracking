#!/bin/bash

mkdir -p datasets/ovis
cd datasets/ovis

declare -A gdrive_ids

gdrive_ids["train"]="11DPalTyrI2OdrDai7pG45gn3W10P1XO5"
gdrive_ids["valid"]="171Z_pqcVkbJQ5lDMVM_mSfscjOPLBW7Z"
gdrive_ids["train_annotations"]="1gIPS-LGJLNhWc4_QGBTGAcSDMtVOaixk"
gdrive_ids["valid_annotations"]="165KJVcFDjiW75PlEqpMcJhG2ChEz6E5w"
gdrive_ids["test_annotations"]="1Oj3HjLkDA7McZ0KjigUzu9dc47Nx5F7Q"

get_dataset() {
    if [ -d "$1" ]
    then
        echo "Directory $1 already exists. Skipping."
    else
        gdown -c "${gdrive_ids[$1]}"
        unzip "$1.zip"
        rm "$1.zip"
    fi
}

gdown -c ${gdrive_ids["train_annotations"]}
gdown -c ${gdrive_ids["valid_annotations"]}
gdown -c ${gdrive_ids["test_annotations"]}
get_dataset "train"
get_dataset "valid"
