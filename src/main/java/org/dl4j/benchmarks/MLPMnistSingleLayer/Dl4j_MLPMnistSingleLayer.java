package org.dl4j.benchmarks.MLPMnistSingleLayer;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.dl4j.benchmarks.Utils.BenchmarkUtil;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 *
 */
public class Dl4j_MLPMnistSingleLayer{
    private static Logger log = LoggerFactory.getLogger(Dl4j_MLPMnistSingleLayer.class);
    protected static final int height = 28;
    protected static final int width = 28;
    public final static int channels = 1;
    public final static int numLabels = 10;
    public final static int batchSize = 128;
    public final static int epochs = 15;
    public final static int iterations = 1;
    public final static int seed = 42;
    public final static int nCores = 32;

    // Multiple GPUs
    public final static boolean multiGPU = false;
    public final static int buffer = 8;
    public final static int workers = 8;
    public final static int avgFrequency = 3;

    public static void main(String[] args) throws Exception {
            long totalTime = System.currentTimeMillis();
//            CudaEnvironment.getInstance().getConfiguration()
//                    .setFirstMemory(AllocationStatus.DEVICE)
//                    .setExecutionModel(Configuration.ExecutionModel.SEQUENTIAL)
//                    .setAllocationModel(Configuration.AllocationModel.CACHE_ALL)
//                    .setMaximumBlockSize(128)
//                    .enableDebug(false)
//                    .setVerbose(false);


            int hiddenNodes = 1000;
            double learningRate = 6e-3;
            double momentum = 0.9;
            double l2 = 1e-4;

            //Get the DataSetIterators:
            long dataLoadTime = System.currentTimeMillis();
            DataSetIterator mnistTrain = new MultipleEpochsIterator(epochs, new MnistDataSetIterator(batchSize,true,12345));
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);
            dataLoadTime = System.currentTimeMillis() - dataLoadTime;

//            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .activation("relu")
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .iterations(1)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(momentum)
                    .regularization(true).l2(l2)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(height * width)
                            .nOut(hiddenNodes)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(hiddenNodes)
                            .nOut(numLabels)
                            .activation("softmax")
                            .build())
                    .pretrain(false).backprop(true)
                    .build();

            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();

//            log.info("Train model....");
            long trainTime = System.currentTimeMillis();
            if(multiGPU) {
                CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).allowCrossDeviceAccess(true);
                ParallelWrapper wrapper = BenchmarkUtil.multiGPUModel(network, buffer, workers, avgFrequency);
                wrapper.fit(mnistTrain);
            } else {
                network.fit(mnistTrain);
            }
            trainTime = System.currentTimeMillis() - trainTime;

//            log.info("Evaluate model....");
            long testTime = System.currentTimeMillis();
            Evaluation eval = network.evaluate(mnistTest);
            log.info(eval.stats());
            testTime = System.currentTimeMillis() - testTime;

            totalTime = System.currentTimeMillis() - totalTime;
            log.info("****************Example finished********************");
            BenchmarkUtil.printTime("Data", dataLoadTime);
            BenchmarkUtil.printTime("Train", trainTime);
            BenchmarkUtil.printTime("Test", testTime);
            BenchmarkUtil.printTime("Total", totalTime);

        }


}

