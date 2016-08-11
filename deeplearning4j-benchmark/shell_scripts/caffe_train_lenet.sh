#!/usr/bin/env sh
# change to -gpu all for multi-gpu

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/CNNMnist/caffe_lenet_solver.prototxt;;
    'GPU' ) time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/CNNMnist/caffe_lenet_solver.prototxt -gpu 0;;
    'MULTI' ) time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/CNNMnist/caffe_lenet_solver.prototxt -gpu all;;
    *) echo "Invalid response";;
esac
