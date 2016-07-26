#!/usr/bin/env sh
# change to -gpu all for multi-gpu

while true; do
    read -p "CPU, GPU, or MULTI?" yn
    case $yn in
        'CPU' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_sigmoid_solver_bn.prototxt;;
        'GPU' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_sigmoid_solver_bn.prototxt -gpu 0;;
        'MULTI' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_sigmoid_solver_bn.prototxt -gpu all;;
        *) echo "Invalid response";;
    esac
done
