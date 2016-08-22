package org.deeplearning4j.ModelCompare.dl4j;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class Dl4j_Lenet {
    protected static int height;
    protected static int width;
    protected static int channels;
    private int numLabels;
    protected double learningRate;
    protected double momentum;
    protected double l2;
    private long seed;

    public Dl4j_Lenet(int height, int width, int channels, int numLabels,
                      double learningRate, double momentum, double l2, long seed) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.l2 = l2;
        this.seed = seed;
    }

    private ConvolutionLayer conv5x5(String name, int nIn, int out) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}).nIn(nIn).nOut(out).name(name).build();
    }

    private SubsamplingLayer maxPool2x2(String name) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2}).name(name).build();
    }

    public MultiLayerNetwork build_model() {
        int ccn1Depth = 20;
        int ccn2Depth = 50;
        int ffn1Depth = 500;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .activation("identity")
                .weightInit(WeightInit.XAVIER)
                .learningRate(learningRate)//.biasLearningRate(2e-2)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true).l2(l2)
                .updater(Updater.NESTEROVS).momentum(momentum)
                .list()
                .layer(0, conv5x5("cnn1", channels, ccn1Depth))
                .layer(1, maxPool2x2("maxpool1"))
                .layer(2, conv5x5("cnn2", 0, ccn2Depth))
                .layer(3, maxPool2x2("maxpool2"))
                .layer(4, new DenseLayer.Builder().name("ffn1").activation("relu").nOut(ffn1Depth).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output").nOut(numLabels).activation("softmax").build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }


}
