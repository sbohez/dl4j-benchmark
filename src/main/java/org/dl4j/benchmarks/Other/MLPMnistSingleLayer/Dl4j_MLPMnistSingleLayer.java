package org.dl4j.benchmarks.Other.MLPMnistSingleLayer;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
            int numEpochs = 15;

            //Get the DataSetIterators:
            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, 10000, true, true, true, rngSeed);
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


            log.info("Build model....");
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

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

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

