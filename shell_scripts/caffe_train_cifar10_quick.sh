#!/usr/bin/env sh
# change to -gpu all for multi-gpu

time caffe train --solver=src/main/java/org/dl4j/benchmarks/Cifar10/caffe_cifar10_full_quick_solver.prototxt -gpu 0