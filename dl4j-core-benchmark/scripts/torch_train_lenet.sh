#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU ' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/CNNMnist/torch-lenet.lua;;
    'GPU' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/CNNMnist/torch-lenet.lua -gpu;;
    'MULTI' ) time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/CNNMnist/torch-lenet.lua -gpu -multi;;
    *) echo "Invalid response";;
esac