#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI?" yn
case $yn in
    'CPU' ) time java -cp target/Dl4j-Benchmarks-1.0-SNAPSHOT.jar org.dl4j.benchmarks.MLPMnistSingleLayer.Dl4j_MLPMnistSingleLayer;;
    'GPU' ) echo "Make sure pom nd4j-backend set to cuda-7.5" && time java -cp target/Dl4j-Benchmarks-1.0-SNAPSHOT.jar org.dl4j.benchmarks.MLPMnistSingleLayer.Dl4j_MLPMnistSingleLayer;;
    'MULTI' ) echo "Make sure pom nd4j-backend set to cuda-7.5" && time java -cp target/Dl4j-Benchmarks-1.0-SNAPSHOT.jar org.dl4j.benchmarks.MLPMnistSingleLayer.Dl4j_MLPMnistSingleLayer -nGW 8;;
    *) echo "Invalid response";;
esac
