#!/usr/bin/env sh

$HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/MLPMnistSingleLayer/caffe_mlp_solver.prototxt

#while true; do
#    read -p "CPU, GPU, or MULTI?" yn
#    case $yn in
#        'CPU' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/MLPMnistSingleLayer/caffe_mlp_solver.prototxt;;
#        'GPU' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/MLPMnistSingleLayer/caffe_mlp_solver.prototxt -gpu 0;;
#        'MULTI' ) time caffe train --solver=src/main/java/org/dl4j/benchmarks/MLPMnistSingleLayer/caffe_mlp_solver.prototxt -gpu all;;
#        *) echo "Invalid response";;
#    esac
#done