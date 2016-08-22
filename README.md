# Dl4j-Benchmarks

Repository to track Dl4j benchmarks in relation to well known frameworks on cpu and gpu and for dl4j on spark.

#### Core Benchmarks

* ModelCompare: compares same structure across all frameworks in this repo
    * MLP: using simple, single layer feed forward with MNIST data 
    * Lenet: using common LeNet CNN model with MNIST data
* Cifar10: compares Dl4j against best structures from each framework 
* Experiment: explores other comparisons and more of storage for drafts and works in progress

#### Spark Benchmarks

The deeplearning4j-spark-benchmark package contains a number of synthetic benchmarks to test Spark training performance under a variety of situations.

For more details, see the readme [here - TODO]


## Core Packages Comparison
Main packages included for comparison so far...

**Dl4j (v0.5.1)**
* Install: http://deeplearning4j.org/quickstart
* Setup packages: add to pom.xml
* Set GPU: change in pom file under nd4j-backends (native for cpu and cuda-7.5 for gpu) 
* Compile: mvn clean install -P (native or cuda)

**Caffe (vr3)**
* Install: http://caffe.berkeleyvision.org/installation.html
* Set GPU: change in solver prototext under solver_mode as either CPU or GPU
* Pass in -gpu # for the number of GPUs to use or all for all

**Tensorflow(v0.9.0)**
* Install: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
* Setup packages: pip install \<*filename*>
* cuDNN required for CNN models and cuDNN > 4 requires to compile from source
* Checkout for configuration fixes: https://stackoverflow.com/questions/37663064/cudnn-compile-configuration-in-tensorflow

**Torch (v7)**
* Install: http://torch.ch/docs/getting-started.html 
* Setup packages: luarocks install \<*filename*>
* Utilize cutorch, cunn, cudnn packages for cuda backend integration 
* cuDNN required for CNN models

To run any of these examples, configure the system to the platform, install this repo and run:

        ./dl4j-core-benchmarks/scripts/model_compare_.sh

Note: If multiple GPUs, control how many used by adding 'export CUDA_VISIBLE_DEVICES=' to .bashrc or .bash_profile and setting empty for CPU and 0,1,2,3 for GPUs (0 if just one and 0,1 if just two) 

#### Benchmark System
Running benchmarks on following system setup:
* Ubuntu 14.0.4
* 60GB RAM 
* 32 Intel Xeon E5-2670 CPUs
* 4 Grid GPUs 4GB RAM
* gcc & g++ v4.9
* BLAS: OpenBLAS v1.13 or Cublas v7.5
* cuDNN v5.1.3

In order to run all examples in core, you need to configure a system for all of the platforms. Each platform differs in requirements and be especially aware of software versions that are not supported.

#### Package Comparisons

Initial analysis. Consider all numbers hostile with potential to change as we get additional reviews sorted.

##### Training Function Timing 

**MLP Example**

| Package    | CPU   | GPU   | Multi | Accuracy |
| ---------- |:-----:| -----:| -----:| --------:| 
| Dl4j       | TBV   | 2m07ms|   48s | ~99.0%   | 
| Caffe      | 2m18s |   13s |   33s | ~97.4%   |
| Tensorflow | 1m10s |   38s | 1m11s | ~98.3%*  |
| Torch      | 4m54s |   51s | 1m34s | ~98.0%   |

**Lenet Example w/ cuDNN**

| Package    | CPU   | GPU   | Multi | Accuracy |
| ---------: |------:| -----:| -----:| --------:| 
| Dl4j       | TBV   | 2m48s |   59s | ~99.0%   | 
| Caffe      | 13m27s|   40s |   55s | ~99.0%   |
| Tensorflow |  5m10s| 1m37s | 2m36s | ~98.6%   |
| Torch      | 17m59s| 6m11s | 3m37s | ~98.3%   |


##### Full Script Timing

**MLP Example**

| Package    | CPU   | GPU   | Multi | Accuracy |
| ---------- |:-----:| -----:| -----:| --------:| 
| Dl4j       | TBV   | 2m16s |   59s | ~97.4%   | 
| Caffe      | 2m20s |   15s |   36s | ~97.4%   |
| Tensorflow | 1m15s |   43s | 1m18s | ~98.3%*  |
| Torch      | 4m56s | 1m03s | 1m46s | ~98.0%   |

**Lenet Example w/ cuDNN**

| Package    | CPU   | GPU   | Multi | Accuracy |
| ---------: |------:| -----:| -----:| --------:| 
| Dl4j       | TVB   | 2m59s | 1m09s | ~99.0%   | 
| Caffe      | 13m31s|   42s |   57s | ~99.0%   |
| Tensorflow | 5m15s | 1m44s | 2m44s | ~98.6%   |
| Torch      | 18m03s| 6m25s | 3m50s | ~98.3%   |

Note: 
 * Tensorflow required learning rate modification on MLP by 1/10th otherwise accuracy drops to 9%
 - Accuracy varies slighty between cpu, single & multi-gpu. 
 - Timings vary (potentially a couple seconds) for all packages on each run
 - Time to transfer and consolidate data can lead to longer performance times on multi-gpu (larger datasets needed for comparison)
 - Issues getting nccl setup on system for Torch multiple gpu tests; thus, not used in tests 

## *How to Help*
Help is welcome to improve comparisons. If you know a better way or see a fix that is needed, please submit a pull request. Top of mind next steps that help would be appreciated:

    - Confirm configurations for all frameworks (seeking outside reviews - esp. on Torch with gpus because timing appears high)
    - Compare LSTMs, Autoencoders, RBMs where available
    - Setup Dl4j AlexNet functionality with multiple GPUs for benchmark
