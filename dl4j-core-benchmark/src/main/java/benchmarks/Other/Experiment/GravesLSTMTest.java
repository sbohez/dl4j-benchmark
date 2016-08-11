package org.dl4j.benchmarks.Other.Experiment;


import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 */

public class GravesLSTMTest {
    int miniBatchSize = 32;
    int nIn = 150;
    int layerSize = 300;
    int timeSeriesLength = 50;
    private int outputNum = 10;
    private long seed = 42;
    private int iterations = 1;
    private int nIterations = 100;
    private int batchSize = 16;
    private int numExamples = 100;
    private MultiLayerNetwork model;

    public void buildModel(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.01)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .nIn(nIn)
                        .nOut(layerSize)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .activation("tanh")
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .nIn(layerSize)
                        .nOut(layerSize)
                        .activation("tanh").build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
                        .nIn(layerSize)
                        .nOut(outputNum).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(timeSeriesLength).tBPTTBackwardLength(timeSeriesLength)
                .build();
        model = new MultiLayerNetwork(conf);
        model.init();

    }

    // TODO put in util
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
        INDArray input = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength});
        model.setInput(input);
        forwardTest(input, model);

    }
}
