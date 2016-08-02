#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI?" yn
case $yn in
    'CPU' ) time th src/main/java/org/dl4j/benchmarks/CNNMnist/torch-lenet.lua;;
    'GPU' ) time th src/main/java/org/dl4j/benchmarks/CNNMnist/torch-lenet.lua -gpu;;
    'MULTI' ) time th src/main/java/org/dl4j/benchmarks/CNNMnist/torch-lenet.lua -gpu -multi;;
    *) echo "Invalid response";;
esac