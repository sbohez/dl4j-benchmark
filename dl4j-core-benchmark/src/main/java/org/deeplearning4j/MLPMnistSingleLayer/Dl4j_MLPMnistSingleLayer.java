package org.deeplearning4j.MLPMnistSingleLayer;

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
import org.deeplearning4j.Utils.BenchmarkUtil;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class Dl4j_MLPMnistSingleLayer{
    private static Logger log = LoggerFactory.getLogger(Dl4j_MLPMnistSingleLayer.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--numGPUWorkers",usage="How many workers to use for multiple GPUs.",aliases = "-nGW")
    // Pass in 8 for 4 GPUs
    public int numGPUWorkers = 0;
    @Option(name="--halfPrecision",usage="Apply half precision for GPUs.",aliases = "-ha")
    public boolean half = false;

    protected final int height = 28;
    protected final int width = 28;
    protected final int channels = 1;
    protected final int numLabels = 10;
    protected final int batchSize = 100;
    protected final int epochs = 15;
    protected final int iterations = 1;
    protected final int seed = 42;
    protected final int nCores = 32; // TODO use for loading data async - there was an issue before
    protected int hiddenNodes = 1000;
    protected double learningRate = 6e-3;
    protected double momentum = 0.9;
    protected double l2 = 1e-4;

    public void run(String[] args) throws Exception {
        long totalTime = System.currentTimeMillis();

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        if(numGPUWorkers > 0 && half)
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
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
        BenchmarkUtil.train(network, numGPUWorkers, mnistTrain);
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

    public static void main(String[] args) throws Exception {
        new Dl4j_MLPMnistSingleLayer().run(args);
    }
}
