package org.dl4j.benchmarks.Other.Experiment;

import org.canova.api.io.filters.BalancedPathFilter;
import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.image.loader.NativeImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Sato's AlexNet benchmark version based on BVLC
 *
 * Sato's code: https://gist.github.com/sato-cloudian/df6386d531d5f724e0a3
 * BVLC: https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/
 */

public class AlexNetBVLC {

    private static final Logger log = LoggerFactory.getLogger(AlexNetBVLC.class);
    private final File trainingFolder;
    private final int numLabels;
    private final double splitTrainTest = 0.8;

    public AlexNetBVLC(File trainingFolder, int numLabels) {
        this.trainingFolder = trainingFolder;
        this.numLabels = numLabels;
    }

    // Note code below creates a list of files but doesn't say number of labels/categories
    private void execute() throws IOException{

        int epochs = 5;
        int height = 227;
        int width = 227;
        int channels = 3;
        int seed = 123;
        int batchSize = 50;
        int iterations = 1;
        int listenerFreq = 1;

        FileSplit fileSplit = new FileSplit(trainingFolder, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        int numExamples =  (int) fileSplit.length();
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(123), new ParentPathLabelGenerator(), numExamples, numLabels, batchSize);

        // Setup train test split
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples*(1+splitTrainTest),  numExamples*(1-splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator(), 255);
        recordReader.initialize(trainData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, dataIter);


        // read images


        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01) // default
                .regularization(true)
                .list()
                .layer(0, new ConvolutionLayer.Builder(11, 11) // 227*227*3 => 55*55*96
                        .nIn(channels)
                        .nOut(96)
                        .padding(0, 0)
                        .stride(4, 4)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3,3}) // 55*55*96 => 27*27*96
                        .padding(0, 0)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5) // 27*27*96 => 27*27*256
                        .nIn(channels)
                        .nOut(256)
                        .padding(2, 2)
                        .stride(1, 1)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3,3}) // 27*27*256 => 13*13*256
                        .padding(0, 0)
                        .stride(2, 2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3) // 13*13*256 => 13*13*384
                        .nIn(channels)
                        .nOut(384)
                        .padding(1, 1)
                        .stride(1, 1)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(5, new ConvolutionLayer.Builder(3, 3) // 13*13*384 => 13*13*384
                        .nIn(channels)
                        .nOut(384)
                        .padding(1, 1)
                        .stride(1, 1)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(6, new ConvolutionLayer.Builder(3, 3) // 13*13*384 => 13*13*256
                        .nIn(channels)
                        .nOut(256)
                        .padding(1, 1)
                        .stride(1, 1)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}) // 13*13*256 => 7*7*256
                        .padding(0, 0)
                        .stride(2, 2)
                        .build())
                .layer(8, new DenseLayer.Builder().activation("relu")
                        .nOut(4096)
                        .dropOut(0.5)
                        .build())
                .layer(9, new DenseLayer.Builder().activation("relu")
                        .nOut(4096)
                        .dropOut(0.5)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
                        .nOut(numLabels)
                        .weightInit(WeightInit.RELU)
                        .activation("softmax")
                        .updater(Updater.SGD)
                        .build())
                .backprop(true).pretrain(false).cnnInputSize(height, width, channels);

        MultiLayerConfiguration conf = builder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        model.fit(trainIter);

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        Evaluation eval = model.evaluate(dataIter);
        log.info(eval.stats());

//        model.setListeners(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(listenerFreq));

        log.info("****************Example finished********************");

    }

    public static void main(String[] args) {

//        AlexNetBVLC alexNetExample = new AlexNetBVLC(new File(args[0])); // Sato's approach to load data

        try {
            AlexNetBVLC alexNetExample = new AlexNetBVLC(new ClassPathResource("ImageExamples").getFile(), 10); // Sato's approach to load data
            alexNetExample.execute();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
