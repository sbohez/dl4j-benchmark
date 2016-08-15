#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/tensorflow_mlp.py ;;
    'GPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/tensorflow_mlp.py --core_type "GPU";;
    'MULTI' ) echo time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/MLPMnistSingleLayer/tensorflow_mlp.py --core_type "MULTI";;
    *) echo "Invalid response";;
esac