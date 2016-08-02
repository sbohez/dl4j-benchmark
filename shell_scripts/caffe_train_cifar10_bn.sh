#!/usr/bin/env sh
# change to -gpu all for multi-gpu

while true; do
    read -p "CPU, GPU, or MULTI?" yn
    case $yn in
        'CPU' ) time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_sigmoid_solver_bn.prototxt;;
        'GPU' ) time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_sigmoid_solver_bn.prototxt -gpu 0;;
        'MULTI' ) time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_sigmoid_solver_bn.prototxt -gpu all;;
        *) echo "Invalid response";;
    esac
done
