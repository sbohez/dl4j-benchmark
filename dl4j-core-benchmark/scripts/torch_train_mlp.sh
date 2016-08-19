#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/torch-main.lua -model_type 'mlp';;
    'GPU' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/torch-main.lua -gpu -model_type 'mlp';;
    'MULTI' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/torch-main.lua -gpu -multi -model_type 'mlp';;
    *) echo "Invalid response";;
esac