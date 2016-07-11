package org.dl4j.benchmarks.Cifar10;

import org.datavec.image.loader.CifarLoader;
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
import org.dl4j.benchmarks.TestModels.CifarCaffeModels;
import org.dl4j.benchmarks.TestModels.CifarModeEnum;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
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
public class Cifar {
    protected static final Logger log = LoggerFactory.getLogger(Cifar.class);
    protected static int HEIGHT = 32;
    protected static int WIDTH = 32;
    protected static int CHANNELS = 3;
    protected static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    protected static int numTestExamples = CifarLoader.NUM_TEST_IMAGES;
    protected static int numLabels = CifarLoader.NUM_LABELS;
    protected static int trainBatchSize;
    protected static int testBatchSize;
    protected static int nCores = 32;

    protected static int seed = 42;
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs;

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

    public static CifarModeEnum networkType = CifarModeEnum.BATCH_NORM;

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network;
        int normalizeValue = 255;

        System.out.println("Load data...");

        log.info("Build model....");
        switch (networkType) {
            case QUICK:
                epochs = 1;
                trainBatchSize = 100;
                testBatchSize = 100;
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
            case FULL_SIGMOID:
                trainBatchSize = 100;
                testBatchSize = 100;
                epochs = 130;
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
            case BATCH_NORM:
                trainBatchSize = 100;
                testBatchSize = 1000;
                epochs = 120;
                nIn = null;
                nOut = new int[]{32, 32, 64};
                activation = "sigmoid";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 12-3;
                biasLearningRate = Double.NaN;
                regularization = false;
                l2 = 0.0;
                momentum = 0.9;
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
        network = new CifarCaffeModels(
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
                momentum).buildNetwork(networkType);


        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        System.out.println("Train model...");
        MultipleEpochsIterator cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{HEIGHT, WIDTH, CHANNELS}, numLabels, null, normalizeValue, true), nCores);
        network.fit(cifar);

        log.info("Evaluate model....");
        CifarDataSetIterator cifarTest = new CifarDataSetIterator(testBatchSize, numTestExamples, new int[] {HEIGHT, WIDTH, CHANNELS}, normalizeValue, false);
        Evaluation eval = network.evaluate(cifarTest);
        System.out.println(eval.stats(true));

        log.info("****************Example finished********************");

        new StandardScaler();

    }

}