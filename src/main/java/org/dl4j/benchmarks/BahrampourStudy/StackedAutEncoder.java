package org.dl4j.benchmarks.BahrampourStudy;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 */
public class StackedAutEncoder {

    private long seed = 123;
    private int outputNum = 10;
    private int iterations = 1;
    private int nIterations = 100;
    private int batchSize = 64;
    private int numExamples = 100;
    private MultiLayerNetwork model;


    public void buildModel() {

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("sigmoid")
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new GaussianDistribution(0, 1))
                .learningRate(1e-3) // TODO check lr
                .biasLearningRate(1e-3) // TODO check lr
                .learningRateScoreBasedDecayRate(1e-1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new AutoEncoder.Builder().lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(batchSize)
                        .nOut(800)
                        .biasInit(0)
                        .build())
                // TODO confirm loss and encoder and decoder nodes
                .layer(1, new AutoEncoder.Builder().lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(800)
                        .nOut(1000)
                        .biasInit(0)
                        .build())
                .layer(2, new AutoEncoder.Builder().lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(1000)
                        .nOut(2000)
                        .biasInit(0)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(2000)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true)
                .pretrain(false);

        model = new MultiLayerNetwork(conf.build());
        model.init();

    }

    public void forwardTest(INDArray input, MultiLayerNetwork model){
        System.out.println("Starting test: (forward pass)");
        long startTime = System.currentTimeMillis();
        for( int i=0; i< nIterations; i++ ){
            model.activate(input);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Total runtime: " + (endTime-startTime)/10000 + "seconds");

    }

    public void backwardTest(INDArray input, MultiLayerNetwork model){
        System.out.println("Starting test: (backward pass)");
        long startTime = System.currentTimeMillis();
        for( int i=0; i< nIterations; i++ ){
            // TODO model.backpropGradient();
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Total runtime: " + (endTime-startTime)/10000 + "seconds");

    }

    public void main() throws Exception{

        buildModel();
        INDArray input = Nd4j.rand(new int[]{1, batchSize});
        model.setInput(input);
        forwardTest(input, model);

    }
}
