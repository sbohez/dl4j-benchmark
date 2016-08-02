#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI?" yn
case $yn in
    'CPU' ) time python src/main/java/org/dl4j/benchmarks/MLPMnistSingleLayer/tensorflow_mlp.py;;
    'GPU' ) time python src/main/java/org/dl4j/benchmarks/MLPMnistSingleLayer/tensorflow_mlp.py -core_type GPU;;
    'MULTI' ) echo "Not implemented";;
    *) echo "Invalid response";;
esac