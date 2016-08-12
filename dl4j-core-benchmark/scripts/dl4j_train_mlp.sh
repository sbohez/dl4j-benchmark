#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.MLPMnistSingleLayer.Dl4j_MLPMnistSingleLayer;;
    'GPU' ) echo "Make sure pom nd4j-backend set to cuda-7.5" && time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.MLPMnistSingleLayer.Dl4j_MLPMnistSingleLayer -P cuda;;
    'MULTI' ) echo "Make sure pom nd4j-backend set to cuda-7.5" && time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.MLPMnistSingleLayer.Dl4j_MLPMnistSingleLayer -P cuda -nGW 8;;
    *) echo "Invalid response";;
esac
