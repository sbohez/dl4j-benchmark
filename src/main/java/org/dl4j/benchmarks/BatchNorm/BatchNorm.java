package org.dl4j.benchmarks.BatchNorm;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 */

public class BatchNorm {

    private static Logger log = LoggerFactory.getLogger(BatchNorm.class);
    final static int numRows = 28;
    final static int numColumns = 28;
    static int outputNum = 10;
    static int batchSize = 128;
    static int rngSeed = 123;
    static int numEpochs = 15;
    static int numExamples = 10000;
    static boolean useBn = true;

    public static MultiLayerNetwork bn(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new BatchNormalization.Builder()
                        .nOut(1000)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork singleLayer(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        return new MultiLayerNetwork(conf);
    }

    public static void main(String[] args) throws Exception {

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, numExamples, false, true, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, numExamples, false, false, true, rngSeed);


        log.info("Build model....");

        MultiLayerNetwork model = useBn? bn(): singleLayer();
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
