package org.dl4j.benchmarks.ModelSpecific;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

    public AlexNetBVLC(File trainingFolder) {
        this.trainingFolder = trainingFolder;
    }

    // Note code below creates a list of files but doesn't say number of labels/categories
    private void execute() throws IOException{

        // create labels
        int samples = 0;
        List<String> labels = new ArrayList<String>();
        for (String labelName : this.trainingFolder.list()) {

            if (new File(this.trainingFolder, labelName).isFile())
                continue;

            System.out.println("adding a label: " + labelName);

            labels.add(labelName);

            File labelFolder = new File(this.trainingFolder, labelName);
            for (String image : labelFolder.list()) {
                if (!image.endsWith("jpg"))
                    continue;
                samples++;
                log.info("added a sample: " + new File(labelFolder, image).getAbsolutePath());
            }

        }

        log.info("outputs, samples = " + labels.size() + ", " + samples);

        // read images
        int width = 227;
        int height = 227;
        int nChannels = 3;
        int seed = 123;

        RecordReader recordReader = new ImageRecordReader(width, height, nChannels, true, labels);
        try{
            recordReader.initialize(new LimitFileSplit(this.trainingFolder, BaseImageLoader.ALLOWED_FORMATS, samples, labels.size(), null, new Random(seed)));
        } catch(InterruptedException ie) {
            ie.printStackTrace();
        }

        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, width * height * nChannels, labels.size());
        iter.setPreProcessor(new ImagePreProcessor());
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        int numRows = height;
        int numColumns = width;
        int outputNum = labels.size();
        int numSamples = samples;
        int batchSize = 50;
        int iterations = 1;
        int nEopcs = 1;
        int listenerFreq = 1;
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01) // default
                .regularization(true)
                .list()
                .layer(0, new ConvolutionLayer.Builder(11, 11) // 227*227*3 => 55*55*96
                        .nIn(nChannels)
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
                        .nIn(nChannels)
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
                        .nIn(nChannels)
                        .nOut(384)
                        .padding(1, 1)
                        .stride(1, 1)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(5, new ConvolutionLayer.Builder(3, 3) // 13*13*384 => 13*13*384
                        .nIn(nChannels)
                        .nOut(384)
                        .padding(1, 1)
                        .stride(1, 1)
                        .weightInit(WeightInit.RELU)
                        .activation("relu")
                        .build())
                .layer(6, new ConvolutionLayer.Builder(3, 3) // 13*13*384 => 13*13*256
                        .nIn(nChannels)
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
                        .nOut(outputNum)
                        .weightInit(WeightInit.RELU)
                        .activation("softmax")
                        .updater(Updater.SGD)
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerConfiguration conf = builder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
//        model.setListeners(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(listenerFreq));
        for(int i=0; i<nEopcs; i++) {
//            while (iter.hasNext()) {
//                DataSet dataSet = iter.next();
//                model.fit(dataSet);
//            }
            model.fit(iter);
            log.info("*** Completed epoch {} ***", i);
            iter.reset();

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(iter.hasNext()) {
                DataSet dataSet = iter.next();
                INDArray output = model.output(dataSet.getFeatureMatrix());
                eval.eval(dataSet.getLabels(), output);
            }
            log.info(eval.stats());
            iter.reset();
        }

        log.info("****************Example finished********************");

    }

    private static class ImagePreProcessor implements DataSetPreProcessor {

        @Override
        public void preProcess(DataSet dataSet) {
            dataSet.getFeatureMatrix().divi(255);  //[0,255] -> [0,1] for input pixel values
        }
    }

    public static void main(String[] args) {

//        AlexNetBVLC alexNetExample = new AlexNetBVLC(new File(args[0])); // Sato's approach to load data

        try {
            AlexNetBVLC alexNetExample = new AlexNetBVLC(new ClassPathResource("ImageExamples").getFile()); // Sato's approach to load data
            alexNetExample.execute();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
