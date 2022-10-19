#!/bin/bash

mkdir -p datasets/coco
cd datasets/coco

declare -A download_paths
declare -A zip_files

download_paths["train2017"]="http://images.cocodataset.org/zips/train2017.zip"
download_paths["val2017"]="http://images.cocodataset.org/zips/val2017.zip"
download_paths["annotations"]="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

zip_files["train2017"]="train2017.zip"
zip_files["val2017"]="val2017.zip"
zip_files["annotations"]="annotations_trainval2017.zip"

get_dataset() {
    if [ -d "$1" ]
    then
        echo "Directory $1 already exists. Skipping."
    else
        axel -ca -n 8 "${download_paths[$1]}"
        unzip "${zip_files[$1]}"
        rm "${zip_files[$1]}"
    fi
}

get_dataset "train2017"
get_dataset "val2017"
get_dataset "annotations"
