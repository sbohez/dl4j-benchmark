package org.dl4j.benchmarks.Other.Standard;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by AlexBlack Aug.
 */
public class GravesLSTMBenchmark {
    public static void main( String[] args ){

        System.out.println("Factory: " + Nd4j.factory());

        int miniBatchSize = 32;
        int nIn = 150;
        int layerSize = 300;
        int timeSeriesLength = 50;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                        .nIn(nIn)
                        .nOut(layerSize)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .activation("tanh")
                        .build())
                .build();

        Layer lstm = LayerFactories.getFactory(conf.getLayer()).create(conf, null, 0, null);

        int nIterationsBefore = 50;
        int nIterations = 100;

        INDArray input = Nd4j.rand(new int[]{miniBatchSize, nIn, timeSeriesLength});
        lstm.setInput(input);

        for( int i=0; i<nIterationsBefore; i++ ){
            //Set input, do a forward pass:
            lstm.activate(true);
            if( i % 50 == 0 ) System.out.println(i);
        }

        System.out.println("Starting test: (forward pass)");
        long startTime = System.currentTimeMillis();
        for( int i=0; i<nIterations; i++ ){
            lstm.activate(true);
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Total runtime: " + (endTime-startTime)/10000 + "seconds");
    }

}
