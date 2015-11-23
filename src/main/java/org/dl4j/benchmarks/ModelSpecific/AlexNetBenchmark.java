package org.dl4j.benchmarks.ModelSpecific;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by nyghtowl on 11/22/15.
 */
public class AlexNetBenchmark {


    public static void main( String[] args ) {
        int height = 100;
        int width = 100;
        int channels = 3;
        int outputNum = 1000;
        int iterations = 90;
        int miniBatchSize = 100;

        MultiLayerNetwork model = AlexNet(height, width, channels, outputNum, 123, iterations);

        int nIterationsBefore = 50;
        int nIterations = 100;

        INDArray input = Nd4j.rand(new int[]{miniBatchSize, channels, height, width});
        model.setInput(input);

        for( int i=0; i<nIterationsBefore; i++ ){
            //Set input, do a forward pass:
            model.activate(true);
            if( i % 50 == 0 ) System.out.println(i);
        }

        System.out.println("Starting test: (forward pass)");
        long startTime = System.currentTimeMillis();
        for( int i=0; i<nIterations; i++ ){
            model.activate(input);
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Total runtime: " + (endTime-startTime)/10000 + "seconds");

    }

    // TODO move AlexNet model into dl4j and reference from there
    public static MultiLayerNetwork AlexNet(int height, int width, int channels, int outputNum, long seed, int iterations) {

        double nonZeroBias = 1;
        double dropOut = 0.5;
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new GaussianDistribution(0.0, 0.01))
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-3)
                .learningRateScoreBasedDecayRate(1e-1)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .miniBatch(false)
                .list(13)
                        //conv1
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder()
                        .name("lrn1")
                        .build())
                .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                        //conv2
                .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                        .name("cnn2")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new LocalResponseNormalization.Builder()
                        .name("lrn2")
                        .k(2).n(5).alpha(1e-4).beta(0.75)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                        //conv3
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(384)
                        .build())
                        //conv4
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                        //conv5
                .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5")
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool3")
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false);


        new ConvolutionLayerSetup(conf,height,width,channels);
        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        return model;
    }

}
