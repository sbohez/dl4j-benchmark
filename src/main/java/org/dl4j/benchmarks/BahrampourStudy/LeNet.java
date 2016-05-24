package org.dl4j.benchmarks.BahrampourStudy;

import net.didion.jwnl.data.Exc;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * https://github.com/DL-Benchmarks/DL-Benchmarks
 */
public class LeNet {

    private int height = 28;
    private int width = 28;
    private int channels = 1;
    private int outputNum = 10;
    private long seed = 123;
    private int iterations = 1;
    private int nIterations = 100;
    private int batchSize = 64;
    private int numExamples = 100;
    private MultiLayerNetwork model;

    public void buildModel(){
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .learningRate(1e-3)
                .biasLearningRate(2e-3)
                .learningRateScoreBasedDecayRate(1e-1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(20)
                        .activation("tahn")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn2")
                        .nOut(50)
                        .activation("tahn")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(500)
                        .build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .activation("softmax") // radial basis function required
                        .build())
                .backprop(true)
                .pretrain(false)
                .cnnInputSize(height,width,channels);


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
        INDArray input = Nd4j.rand(new int[]{batchSize, channels, height, width});
        model.setInput(input);
        forwardTest(input, model);

    }
}
