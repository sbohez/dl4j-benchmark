package org.deeplearning4j.Cifar10.dl4j;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.Utils.DL4J_Utils;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

/**
 * CIFAR-10 is an image dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset inculdes 60K
 * tiny RGB images sized 32 x 32 pixels covering 10 classes. There are 50K training images and 10K test images.
 *
 * Use this example to run cifar-10.
 *
 * Reference: https://www.cs.toronto.edu/~kriz/cifar.html
 * Dataset url: https://s3.amazonaws.com/dl4j-distribution/cifar-small.bin
 * Model: https://gist.github.com/mavenlin/e56253735ef32c3c296d
 *
 */
public class Dl4j_Cifar10 {
    protected static final Logger log = LoggerFactory.getLogger(Dl4j_Cifar10.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type CAFFE_BATCH_NORM, CAFFE_FULL_SIGMOID, CAFFE_QUICK, TENSORFLOW_INFERENCE, TORCH_NIN, TORCH_VGG.", aliases = "-mT")
    public String modelType = "TORCH_VGG";
    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
    public int numGPUs = 0;
    @Option(name="--numTrainExamples",usage="Num train examples.",aliases = "-nTrain")
    public int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    @Option(name="--numTestExamples",usage="Num test examples.",aliases = "-nTest")
    public int numTestExamples = CifarLoader.NUM_TEST_IMAGES;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-nTrainB")
    public int trainBatchSize = 100;
    @Option(name="--testBatchSize",usage="Test batch size.",aliases = "-nTestB")
    public int testBatchSize = 100;
    @Option(name="--epochs",usage="Number of epochs.",aliases = "-epochs")
    public int epochs = 1;
    @Option(name="--preProcess",usage="Set preprocess.",aliases = "-pre")
    public boolean preProcess = false;

    protected static int HEIGHT = 32;
    protected static int WIDTH = 32;
    protected static int CHANNELS = 3;
    protected static int numLabels = CifarLoader.NUM_LABELS;

    protected static int seed = 42;
    protected static int listenerFreq = 1;
    protected static int iterations = 1;

    protected static int[] nIn;
    protected static int[] nOut;
    protected static String activation;
    protected static WeightInit weightInit;
    protected static OptimizationAlgorithm optimizationAlgorithm;
    protected static Updater updater;
    protected static LossFunctions.LossFunction lossFunctions;
    protected static double learningRate;
    protected static double biasLearningRate;
    protected static boolean regularization;
    protected static double l2;
    protected static double momentum;
    protected static MultiLayerNetwork network;
    protected boolean train = true;

    public void setVaribales() {
        switch (CifarModeEnum.valueOf(modelType)) {
            case CAFFE_QUICK:
//                trainBatchSize = 100;
//                testBatchSize = 100;
//                epochs = 1;
                nIn = null;
                nOut = new int[]{32, 32, 64, 64};
                activation = "relu";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-3;
                biasLearningRate = 2e-3;
                regularization = true;
                l2 = 4e-3;
                momentum = 0.9;
                break;
            case CAFFE_FULL_SIGMOID:
//                trainBatchSize = 100;
//                testBatchSize = 100;
//                epochs = 130;
                nIn = null;
                nOut = new int[]{32, 32, 64, 250};
                activation = "relu";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-4;
                biasLearningRate = 2e-4;
                regularization = true;
                l2 = 4e-3;
                momentum = 0.9;
                break;
            case CAFFE_BATCH_NORM:
//                trainBatchSize = 100;
//                testBatchSize = 1000;
//                epochs = 120;
                nIn = null;
                nOut = new int[]{32, 32, 64};
                activation = "sigmoid"; // TODO confirm sigmoid matches caffe sigmoid1
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-3;
                biasLearningRate = Double.NaN;
                regularization = false;
                l2 = 0.0;
                momentum = 0.9;
                break;
            case TENSORFLOW_INFERENCE:
//                trainBatchSize = 128;
//                testBatchSize = 128;
//                epochs = 768;
                nIn = null;
                nOut = new int[]{64, 64, 384, 192};
                activation = "relu";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.SGD;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-1;
                biasLearningRate = Double.NaN;
                regularization = true;
                l2 = 0;
                momentum = 0.9;
                break;
            // TODO double check learning rate policy
            case TORCH_NIN:
//                trainBatchSize = 128;
//                testBatchSize = 128;
//                epochs = 300;
                nIn = null;
                nOut = new int[]{192,160,96,192,192,192,192,192,192,10};
                activation = "relu";
                weightInit = WeightInit.RELU;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.SGD;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-1;
                biasLearningRate = Double.NaN;
                regularization = true;
                l2 = 5e-4;
                momentum = 0.9;
                break;
            case TORCH_VGG:
//                trainBatchSize = 128;
//                testBatchSize = 128;
//                epochs = 300;
                nIn = null;
                nOut = new int[]{64,64,128,128,256,256,256,512,512,512,512,512,512,512,512};
                activation = "relu";
                weightInit = WeightInit.RELU;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.SGD;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1;
                biasLearningRate = Double.NaN;
                regularization = true;
                l2 = 5e-4;
                momentum = 0.9;
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
    }

    public void run(String[] args) throws IOException {
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

        setVaribales();

        log.debug("Load data...");
        ImageTransform flip = new FlipImageTransform(seed); // Should random flip some images but not all
        CifarDataSetIterator cifar = new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{HEIGHT, WIDTH, CHANNELS}, numLabels, flip, preProcess, train);
        MultipleEpochsIterator iter = new MultipleEpochsIterator(epochs, cifar);

        log.debug("Build model....");
        network = new CifarModels(
                HEIGHT,
                WIDTH,
                CHANNELS,
                numLabels,
                seed,
                iterations,
                nIn,
                nOut,
                activation,
                weightInit,
                optimizationAlgorithm,
                updater,
                lossFunctions,
                learningRate,
                biasLearningRate,
                regularization,
                l2,
                momentum).buildNetwork(CifarModeEnum.valueOf(modelType));
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.debug("Train model...");
        long dataLoadTime = System.currentTimeMillis();
        dataLoadTime = System.currentTimeMillis() - dataLoadTime;

        long trainTime = System.currentTimeMillis();
        DL4J_Utils.train(network, iter, numGPUs);
        trainTime = System.currentTimeMillis() - trainTime;

        log.info("Evaluate model....");
        long testTime = System.currentTimeMillis();
        cifar.test(numTestExamples, testBatchSize);
        epochs = 1;
        iter = new MultipleEpochsIterator(epochs, cifar);
        Evaluation eval = network.evaluate(iter);
        log.debug(eval.stats(true));
        DecimalFormat df = new DecimalFormat("#.####");
        log.info(df.format(eval.accuracy()));
        testTime =  System.currentTimeMillis() - testTime;
        totalTime = System.currentTimeMillis() - totalTime;

        log.info("****************Example finished********************");
        DL4J_Utils.printTime("Data", dataLoadTime);
        DL4J_Utils.printTime("Train", trainTime);
        DL4J_Utils.printTime("Test", testTime);
        DL4J_Utils.printTime("Total", totalTime);

    }

    public static void main(String[] args) throws Exception {
        new Dl4j_Cifar10().run(args);
    }

}