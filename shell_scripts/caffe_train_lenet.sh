#!/usr/bin/env sh
# change to -gpu all for multi-gpu

while true; do
    read -p "CPU, GPU, or MULTI?" yn
    case $yn in
        'CPU' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/CNNMnist/caffe_lenet_solver.prototxt;;
        'GPU' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/CNNMnist/caffe_lenet_solver.prototxt -gpu 0;;
        'MULTI' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/CNNMnist/caffe_lenet_solver.prototxt -gpu all;;
        *) echo "Invalid response";;
    esac
done
