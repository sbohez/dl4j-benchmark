# Dl4j-Benchmarks

Repository to track Dl4j benchmarks with comparisons internally and against other well known frameworks. Also setup to track performance on cpu and gpu. Note CNNs are run with cuDNN.  

#### Examples
    - MLP: match all platforms to one model structure
    - Lenet: match all platforms to one model structure
    - Cifar10: different platform examples and match different dl4j structures for comparison 
    - Other: experiments still being shaped


## Packages
Main packages included for comparison so far...

**Caffe (vr3)**
    - Install: http://caffe.berkeleyvision.org/installation.html
    - Tricky to complete successfully based on gcc version, BLAS and cuda
    - Cmd to run: sh \<*train bash file*>

**Dl4j (v0.4.1)**
    - Install: http://deeplearning4j.org/quickstart
    - Setup packages: add to pom.xml
    - Cmd to run: mvn clean install && java -cp target/Dl4j-Benchmarks-0.4.0-SNAPSHOT.jar \<*class path*>

**Tensorflow(v0.9.0)**
    - Install: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
    - Setup packages: pip install \<*filename*>
    - Cmd to run: python \<*filename*>
    - Note required to compile from source if using cuDNN > 4

**Torch (v7)**
    - Install: http://torch.ch/docs/getting-started.html 
    - Setup packages: luarocks install \<*filename*>
    - Cmd to run: th \<*filename*>

##  CPU vs GPU Switch 
**Caffe**
    - Change flag in solver prototext
    - Pass in -gpu # for the number of GPUs to use or all for all
    
**Dl4j**
    - Change pom.xml file backend between nd4j-native and nd4j-cuda-7.5

**TensforFlow**:
    - Set config = tf.ConfigProto(device_count={'GPU': #}) and pass into the session
        - replace # with 0 for CPU and 1 or more for GPU
    - cuDNN required for CNN models
    - Checkout for configuration fixes: https://stackoverflow.com/questions/37663064/cudnn-compile-configuration-in-tensorflow

**Torch**
    - Utilize cutorch, cunn, cudnn packages for cuda backend integration 

If multiple GPUs, control how many used by adding 'export CUDA_VISIBLE_DEVICES=' to .bashrc or .bash_profile and setting empty for CPU and 0,1,2,3 for GPUs (0 if just one and 0,1 if just two) 
    
## Benchmark Setup 
Running benchmarks on following system setup:
    - Ubuntu 14.0.4
    - 60GB RAM 
    - 32 Intel Xeon E5-2670 CPUs
    - 4 Grid GPUs 4GB RAM
    - gcc & g++ v4.9
    - BLAS: OpenBLAS v1.13 or Cublas v7.5
    - cuDNN v5.1.3
  
  Running benchmarks on following system setup:
      - Ubuntu 16.0.4
      - 60GB RAM 
      - 32 Intel Xeon E5-2670 CPUs
      - 4 Grid GPUs 4GB RAM
      - gcc & g++ v5.4.1
      - BLAS: OpenBLAS v1.13 or Cublas v7.5
      - cuDNN v5.1.3