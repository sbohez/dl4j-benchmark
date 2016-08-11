#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/tensorflow_mlp.py "CPU";;
    'GPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/tensorflow_mlp.py "GPU";;
    'MULTI' ) echo "Not implemented";;
    *) echo "Invalid response";;
esac