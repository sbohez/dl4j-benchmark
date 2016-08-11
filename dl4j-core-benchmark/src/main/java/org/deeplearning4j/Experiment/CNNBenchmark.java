package org.deeplearning4j.Experiment;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Basic CNN Layer Benchmark
 */
public class CNNBenchmark {

    public static void main( String[] args ){

        int numRows = 28;
        int numColumns = 28;
        int channel = 3;
        int miniBatchSize = 128;
        int kernels = 100;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder()
                        .nIn(channel)
                        .nOut(kernels)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .activation("relu")
                        .kernelSize(2,2)
                        .stride(1,1)
                        .padding(1,1)
                        .build())
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);

        Layer cnn = LayerFactories.getFactory(conf.getLayer()).create(conf, null, 0, params, true);

        int nIterationsBefore = 50;
        int nIterations = 100;

        INDArray input = Nd4j.rand(new int[]{miniBatchSize, channel, numRows, numColumns});
        cnn.setInput(input);

        for( int i=0; i<nIterationsBefore; i++ ){
            //Set input, do a forward pass:
            cnn.activate(true);
            if( i % 50 == 0 ) System.out.println(i);
        }

        System.out.println("Starting test: (forward pass)");
        long startTime = System.currentTimeMillis();
        for( int i=0; i<nIterations; i++ ){
            cnn.activate(input);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Total runtime: " + (endTime-startTime) + " milliseconds");
    }
}
