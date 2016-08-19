#!/usr/bin/env bash

read -p "CPU, GPU, or MULTI? " yn
case $yn in
    'CPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/tensorflow_main.py --model_type "mlp";;
    'GPU' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/tensorflow_main.py --core_type "GPU" --num_gpus 1 --model_type "mlp";;
    'MULTI' ) time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/tensorflow_main.py --core_type "MULTI" --num_gpus 4 --model_type "mlp";;
    *) echo "Invalid response";;
esac