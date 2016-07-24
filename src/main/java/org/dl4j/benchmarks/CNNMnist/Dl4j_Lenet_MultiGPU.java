package org.dl4j.benchmarks.CNNMnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.dl4j.benchmarks.Models.LeNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by @raver119
 */
public class Dl4j_Lenet_MultiGPU extends Dl4j_LenetMnist {

    private static final Logger log = LoggerFactory.getLogger(Dl4j_Lenet_MultiGPU.class);

    public static void main(String[] args) throws Exception {
        long duration = System.currentTimeMillis();
        DataSetIterator mnistTrain = new MnistDataSetIterator(trainBatchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(testBatchSize, false, 12345);

        MultiLayerNetwork network = new LeNet(height, width, channels, numLabels, seed, iterations).init();
        network.init();

        log.info("Train model....");
        network.setListeners(new ScoreIterationListener(100));

        /// Setup to run on multi-gpus ////
        ParallelWrapper wrapper = new ParallelWrapper.Builder(network)
                .prefetchBuffer(8)
                .workers(4)
                .averagingFrequency(3)
                .build();

        for(int i=0; i < epochs; i++) {
            wrapper.fit(mnistTrain);
            if (i != epochs-1) mnistTrain.reset();
        }

        log.info("Evaluate model....");
        Evaluation eval = network.evaluate(mnistTest);
        log.info(eval.stats());
        log.info("****************Example finished********************");
        log.info("Total time: {}", (System.currentTimeMillis() - duration));
    }
}