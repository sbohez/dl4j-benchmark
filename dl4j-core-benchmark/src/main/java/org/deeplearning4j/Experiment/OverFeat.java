package org.deeplearning4j.Experiment;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class OverFeat {

    private int height = 231;
    private int width = 231;
    private int channels = 3;
    private int numLabels = 1000;
    private long seed = 42;
    private int iterations = 90;

    public OverFeat() {}

    public OverFeat(int height, int width, int channels, int numLabels, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerConfiguration conf() {
        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(poolingType, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(poolingType, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(512)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(1024)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(1024)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(poolingType, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool3")
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(3072)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .cnnInputSize(height,width,channels);

        return conf.build();
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

}
