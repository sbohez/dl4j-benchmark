#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time python dl4j-core-benchmark/src/main/java/org/dl4j/benchmarks/CNNMnist/tensorflow_lenet.py "CPU";;
    'GPU' ) time python dl4j-core-benchmark/src/main/java/org/dl4j/benchmarks/CNNMnist/tensorflow_lenet.py "GPU";;
    'MULTI' ) time python dl4j-core-benchmark/src/main/java/org/dl4j/benchmarks/CNNMnist/tensorflow_lenet.py "MULTI";;
    *) echo "Invalid response";;
esac