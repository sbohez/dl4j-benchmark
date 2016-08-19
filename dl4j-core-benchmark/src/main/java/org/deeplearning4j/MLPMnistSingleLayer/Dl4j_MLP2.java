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
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

/**
 */
public class Dl4j_MLP2 {
    private static Logger log = LoggerFactory.getLogger(Dl4j_MLP2.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--numGPUWorkers",usage="How many workers to use for multiple GPUs.",aliases = "-nGPU")
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

    public final static int buffer = 24;
    public final static int avgFrequency = 3;

    public void printTime(String name, long ms){
        log.info(name + " time: {} min, {} sec | {} milliseconds",
                TimeUnit.MILLISECONDS.toMinutes(ms),
                TimeUnit.MILLISECONDS.toSeconds(ms) -
                        TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(ms)),
                ms);

    }

    public ParallelWrapper multiGPUModel(MultiLayerNetwork network, int buffer, int workers, int avgFrequency) {
        return new ParallelWrapper.Builder(network)
                .prefetchBuffer(buffer)
                .workers(workers)
                .averagingFrequency(avgFrequency)
                .build();
    }

    public void train(MultiLayerNetwork network, int numGPUWorkers, DataSetIterator data) {
        org.nd4j.jita.conf.CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true)
                .setMaximumDeviceCache(4L * 1024L * 1024L).setMaximumHostCache(8L * 1024L * 1024L)
                .setMaximumGridSize(512).setMaximumBlockSize(512).allowCrossDeviceAccess(true);
        if (numGPUWorkers > 0) {
            ParallelWrapper wrapper = multiGPUModel(network, buffer, numGPUWorkers, avgFrequency);
            wrapper.fit(data);
        } else {
            network.fit(data);
        }
    }

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
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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
        train(network, numGPUWorkers, mnistTrain);
        trainTime = System.currentTimeMillis() - trainTime;

//            log.info("Evaluate model....");
        long testTime = System.currentTimeMillis();
        Evaluation eval = network.evaluate(mnistTest);
        log.debug(eval.stats());
        DecimalFormat df = new DecimalFormat("#.####");
        log.info(df.format(eval.accuracy()));
        testTime = System.currentTimeMillis() - testTime;

        totalTime = System.currentTimeMillis() - totalTime;
        log.info("****************Example finished********************");
        printTime("Data", dataLoadTime);
        printTime("Train", trainTime);
        printTime("Test", testTime);
        printTime("Total", totalTime);

    }

    public static void main(String[] args) throws Exception {
        new Dl4j_MLPMnistSingleLayer().run(args);
    }
}
