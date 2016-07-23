# Dl4j-Benchmarks

Repository to track Dl4j benchmarks. Work in progress.

## Examples
    - MLP
    - Lenet
    - Cifar10

## Run from CLI

Caffe: sh \<*train bash file*>
Dl4j: mvn clean install && java -cp target/Dl4j-Benchmarks-0.4.0-SNAPSHOT.jar \<*class path*> 
TensorFlow: python \<*filename*>
Torch: th \<*filename*>

## Packages
Caffe (vr3)
    - Install: http://caffe.berkeleyvision.org/installation.html
    - Tricky to complete successfully based on gcc version, BLAS and cuda

Dl4j (v0.4.1)
    - Install: http://deeplearning4j.org/quickstart
    - Setup packages: add to pom.xml

Tensorflow(v0.9.0)
    - Install: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
    - Setup packages: pip install \<*filename*>

Torch (v7)
    - Install: http://torch.ch/docs/getting-started.html 
    - Setup packages: luarocks install \<*filename*>

## Switch CPU vs GPU 
Caffe: 
    - Change flag in solver prototext
    
Dl4j
    - Change pom.xml file backend between nd4j-native and nd4j-cuda-7.5

TensforFlow:
    - put the following in .bashrc or .bash_profile: export CUDA_VISIBLE_DEVICES=
        - empty for CPU and numbers for number of GPUs
    - set config = tf.ConfigProto(device_count={'GPU': #}) and pass into the session
        - replace # with 0 for CPU and 1 or more for GPU
    - cuDNN required for CNN models
    - Checkout for configuration fixes: https://stackoverflow.com/questions/37663064/cudnn-compile-configuration-in-tensorflow

Torch
    - put the following in .bashrc or .bash_profile: export CUDA_VISIBLE_DEVICES=
        - empty for CPU and numbers for number of GPUs

    
## Test Setup 
Running benchmarkcs on following system setup
    - Ubuntu 14.0.4
    - 60GB RAM 
    - 32 Intel Xeon E5-2670 CPUs
    - 4 Grid GPUs 4GB RAM
    - gcc & g++ v4.9
    - BLAS: OpenBLAS v1.13 or Cublas v7.5
    - cuDNN v5
  