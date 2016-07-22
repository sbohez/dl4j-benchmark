# Dl4j-Benchmarks

This project is dedicated to tracking benchmark code for DL4J.

### Run from CLI

Caffe: sh <train bash file>
DL4J: mvn clean install && java -cp target/Dl4j-Benchmarks-0.4.0-SNAPSHOT.jar <class path> 
TensorFlow: python <filename>
Torch: th <filename>

### Packages

Torch
    - Install packages: luarocks install <package>

### CPU vs GPU 
Caffe: 
    - Change flag in solver prototext

TensforFlow:
    - set following empty for CPU in .bashrc or .bash_profile: export CUDA_VISIBLE_DEVICES=
    - set config = tf.ConfigProto(device_count={'GPU': 0}) and pass into the session
        - 0 for cpu and 1 or more for gpu

Torch
    - set following to 0 for CPU and 1 or more for GPU: CUDA_VISIBLE_DEVICES=0 

### cuDNN

TensforFlow:
    - Requires for CNN models to work on GPU
    - Checkout for configuration fixes: https://stackoverflow.com/questions/37663064/cudnn-compile-configuration-in-tensorflow
    
  