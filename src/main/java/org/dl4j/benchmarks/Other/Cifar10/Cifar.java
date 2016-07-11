package org.dl4j.benchmarks.Other.Cifar10;

import org.canova.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.dl4j.benchmarks.Other.Cifar10.TestModels.CifarBatchNorm;
import org.dl4j.benchmarks.Other.Cifar10.TestModels.CifarFullSigmoid;
import org.dl4j.benchmarks.Other.Cifar10.TestModels.CifarQuickModel;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
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
    protected static int height = 32;
    protected static int width = 32;
    protected static int channels = 3;
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

    public static String model = "Norm";

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network;
        int normalizeValue = 255;

        System.out.println("Load data...");

        log.info("Build model....");
        switch (model) {
            case "Quick":
                trainBatchSize = 100;
                testBatchSize = 100;
                epochs = 1;
                network = new CifarQuickModel(height, width, channels, numLabels, seed, iterations).init();
                break;
            case "FullSigmoid":
                trainBatchSize = 100;
                testBatchSize = 100;
                epochs = 130;
                network = new CifarFullSigmoid(height, width, channels, numLabels, seed, iterations).init();
                break;
            case "Norm":
                trainBatchSize = 100;
                testBatchSize = 1000;
                epochs = 120;
                network = new CifarBatchNorm(height, width, channels, numLabels, seed, iterations).init();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");

        }

        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        System.out.println("Train model...");
        MultipleEpochsIterator cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(trainBatchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, null, normalizeValue, true), nCores);
        network.fit(cifar);

        log.info("Evaluate model....");
        CifarDataSetIterator cifarTest = new CifarDataSetIterator(testBatchSize, numTestExamples, new int[] {height, width, channels}, normalizeValue, false);
        Evaluation eval = network.evaluate(cifarTest);
        System.out.println(eval.stats(true));

        log.info("****************Example finished********************");

        new StandardScaler();

    }

}