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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class Dl4j_MLPMnistSingleLayer{
        private static Logger log = LoggerFactory.getLogger(Dl4j_MLPMnistSingleLayer.class);

        public static void main(String[] args) throws Exception {

//            CudaEnvironment.getInstance().getConfiguration()
//                    .setFirstMemory(AllocationStatus.DEVICE)
//                    .setExecutionModel(Configuration.ExecutionModel.SEQUENTIAL)
//                    .setAllocationModel(Configuration.AllocationModel.CACHE_ALL)
//                    .setMaximumBlockSize(128)
//                    .enableDebug(false)
//                    .setVerbose(false);

            final int numRows = 28;
            final int numColumns = 28;
            int outputNum = 10;
            int batchSize = 128;
            int rngSeed = 123;
            int epochs = 15;
            int hiddenNodes = 1000;
            double learningRate = 6e-4;
            double momentum = 0.9;
            double l2 = 1e-4;
            
            long duration = System.currentTimeMillis();

            //Get the DataSetIterators:
            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(rngSeed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .iterations(1)
                    .learningRate(learningRate)
                    .updater(Updater.NESTEROVS).momentum(momentum)
                    .regularization(true).l2(l2)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(numRows * numColumns)
                            .nOut(hiddenNodes)
                            .activation("relu")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(hiddenNodes)
                            .nOut(outputNum)
                            .activation("softmax")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .pretrain(false).backprop(true)
                    .build();

            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();

            log.info("Train model....");

            for(int i=0; i < epochs; i++)
                network.fit(mnistTrain);

            log.info("Evaluate model....");
            Evaluation eval = network.evaluate(mnistTest);

            log.info(eval.stats());
            log.info("****************Example finished********************");
            log.info("Total time: {}", (System.currentTimeMillis() - duration));


        }

}

