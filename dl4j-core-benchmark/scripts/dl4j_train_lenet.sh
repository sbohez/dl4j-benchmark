#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.CNNMnist.Dl4j_LenetMnist;;
    'GPU' ) echo "Make sure pom nd4j-backend set to cuda-7.5" && time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.CNNMnist.Dl4j_LenetMnist;;
    'MULTI' ) echo "Make sure pom nd4j-backend set to cuda-7.5" && time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.CNNMnist.Dl4j_LenetMnist -nGPU 12;;
     *) echo "Invalid response";;
esac
