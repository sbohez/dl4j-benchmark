#!/usr/bin/env sh
# change to -gpu all for multi-gpu

time $HOME/caffe/build/tools/caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_solver_lr2.prototxt -gpu 0