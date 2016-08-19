#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU ' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Util/torch-main.lua --model_type 'lenet';;
    'GPU' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Util/torch-main.lua -gpu --model_type 'lenet';;
    'MULTI' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Util/torch-main.lua -gpu -multi --model_type 'lenet';;
    *) echo "Invalid response";;
esac