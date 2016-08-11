#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/torch-mlp.lua;;
    'GPU' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/torch-mlp.lua -gpu;;
    'MULTI' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/torch-mlp.lua -gpu -multi;;
    *) echo "Invalid response";;
esac