package org.deeplearning4j.Experiment;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Version based on original paper and Caffe
 *
 * Note not all functionality in Caffe exists in Dl4j currently
 *
 */
public class AlexNetBenchmark {


    public static void main( String[] args ) {
        int height = 100;
        int width = 100;
        int channels = 3;
        int outputNum = 10;
        int iterations = 1;
        int miniBatchSize = 50;

        MultiLayerNetwork model = new AlexNet(height, width, channels, outputNum, 123, iterations).init();

        int nIterationsBefore = 1;
        int nIterations = 1;

        INDArray input = Nd4j.rand(new int[]{miniBatchSize, channels, height, width});
        model.setInput(input);

//        for( int i=0; i<nIterationsBefore; i++ ){
//            //Set input, do a forward pass:
//            model.feedForward(true);
//            if( i % 50 == 0 ) System.out.println(i);
//        }


        // TODO setup with data that enables mult iterations on feedForward and get the intial setup back in
        System.out.println("Starting test: (forward pass)");
        long startTime = System.currentTimeMillis();
        for( int i=0; i<nIterations; i++ )
            model.feedForward(true);

        long endTime = System.currentTimeMillis();
        System.out.println("Total runtime: " + (endTime-startTime) + "milliseconds");

    }

}
