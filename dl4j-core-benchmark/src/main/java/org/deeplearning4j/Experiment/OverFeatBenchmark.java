package org.deeplearning4j.Experiment;

import java.util.ArrayList;
import java.util.List;
import java.lang.Math;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Version based on original paper and Caffe
 *
 * Note not all functionality in Caffe exists in Dl4j currently
 *
 */
public class OverFeatBenchmark {


    public static void main( String[] args ) {
        int miniBatchSize = 128;

        if(args.length > 0) {
            miniBatchSize = Integer.parseInt(args[0]);
        }

        MultiLayerNetwork model = new OverFeat().init();

        int nIterationsBefore = 10;
        int nIterations = 1000;

        INDArray input = Nd4j.rand(new int[]{miniBatchSize, 3, 231, 231});
        model.setInput(input);

        System.out.println("Warming up...");
        for( int i=0; i<nIterationsBefore; i++ )
            model.feedForward(true);

        System.out.println("Starting test: (forward pass)");
        List<Double> timings = new ArrayList<>();
        for( int i=0; i<nIterations; i++ ){
            if(i%10==0)
                System.out.println("Iteration "+i);

            long t1 = System.nanoTime();
            //Set input, do a forward pass:
            model.feedForward(true);
            long t2 = System.nanoTime();
//          System.out.println((t2-t1)/1e6);
            timings.add((t2-t1)/1e6);
        }

        double mean = 0;

        for(Double d : timings)
             mean += d;

        mean /= timings.size();

        double stdev = 0;

        for(Double d : timings)
             stdev += (d-mean)*(d-mean);

        stdev /= timings.size()-1;

        stdev = Math.sqrt(stdev);

        System.out.println("Mean: " + mean + " ms");
        System.out.println("Stdev: " + stdev + " ms");
    }

}
