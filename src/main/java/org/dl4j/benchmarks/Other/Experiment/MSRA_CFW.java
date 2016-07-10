package org.dl4j.benchmarks.Other.Experiment;

import org.canova.api.io.filters.BalancedPathFilter;
import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.loader.NativeImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

/**
 * WORK IN PROGRESS Face Classification
 *
 * Microsoft Research MSRA-CFW Celebrity Faces Dataset
 *
 * This is an image classification example built from scratch. You can swap out your own dataset with this structure.
 * Note additional work is needed to build out the structure to work with your dataset.
 * Dataset:
 *      - Celebrity Faces created by MicrosoftResearch
 *      - Based on thumbnails data set which is a smaller subset
 *      - 2215 images & 10 classifications with each image only including one face
 *      - Dataset has more examples of a each person than LFW to make standard classification approaches appropriate
 *      - Gender variation is something built separate from the dataset
 *
 * Checkout this link for more information and to access data: http://research.microsoft.com/en-us/projects/msra-cfw/
 */

public class MSRA_CFW {
    private static final Logger log = LoggerFactory.getLogger(MSRA_CFW.class);

    // based on small sample
    public final static int NUM_IMAGES = 2215; // # examples per person range 50 to 700
    public final static int NUM_LABELS = 10;
    public final static int HEIGHT = 100;
    public final static int WIDTH = 100; // size varies
    public final static int CHANNELS = 3;

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--numExamples",usage="Number of examples",aliases="-nE")
    protected int numExamples = 100;
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 50;
    @Option(name="--epochs",usage="Number of epochs",aliases="-ep")
    protected int epochs = 5;
    @Option(name="--iter",usage="Number of iterations",aliases="-i")
    protected int iterations = 1;
    @Option(name="--numLabels",usage="Number of categories",aliases="-nL")
    protected int numLabels = 2;

    @Option(name="--weightInit",usage="How to initialize weights",aliases="-wI")
    protected WeightInit weightInit = WeightInit.XAVIER;
    @Option(name="--activation",usage="Activation function to use",aliases="-a")
    protected String activation = "relu";
    @Option(name="--updater",usage="Updater to apply gradient changes",aliases="-up")
    protected Updater updater = Updater.NESTEROVS;
    @Option(name="--learningRate", usage="Learning rate", aliases="-lr")
    protected double lr = 1e-3;
    @Option(name="--momentum",usage="Momentum rate",aliases="-mu")
    protected double mu = 0.9;
    @Option(name="--lambda",usage="L2 weight decay",aliases="-l2")
    protected double l2 = 5e-4;
    @Option(name="--regularization",usage="Boolean to apply regularization",aliases="-reg")
    protected boolean regularization = true;

    protected SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;
    protected double nonZeroBias = 1;
    protected double dropOut = 0.5;
    protected double splitTrainTest = 0.8;

    public void run(String[] args) throws Exception{
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        int seed = 123;
        int listenerFreq = 1;


        log.info("Load data....");
        File mainPath;
        int numLabels;
        boolean gender = false;
        if(gender) {
            numLabels = 2;
            mainPath = new File(BaseImageLoader.BASE_DIR, "gender_class");
        }else{
            numLabels = 10;
//            mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample"); // 10 labels
            mainPath = new File(BaseImageLoader.BASE_DIR, "data/mrsa-cfw"); // 10 labels

        }
        // Organize  & limit data file paths
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(123), new ParentPathLabelGenerator(), numExamples, numLabels, batchSize);

        // Setup train test split
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples*(1+splitTrainTest),  numExamples*(1-splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, new ParentPathLabelGenerator(), 255);
        recordReader.initialize(trainData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(lr)
                .momentum(mu)
                .regularization(regularization)
                .l2(l2)
                .updater(updater)
                .useDropConnect(true)

                // AlexNet
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3})
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder()
                        .name("lrn1")
                        .build())
                .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
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
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3")
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4")
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
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
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .dropOut(dropOut)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation("softmax")
                        .build())

                // TensorFlow
//                .list()
//                .layer(0, new ConvolutionLayer.Builder(5, 5)
//                        .name("cnn1")
//                        .nIn(CHANNELS)
//                        .dist(new NormalDistribution(0, 1e-4))
//                        .stride(1, 1)
//                        .padding(2, 2)
//                        .nOut(64)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
//                        .name("pool1")
//                        .build())
//                .layer(2, new LocalResponseNormalization.Builder(4, 1e-3/9, 0.75).build())
//                .layer(3, new ConvolutionLayer.Builder(5, 5)
//                        .name("cnn2")
//                        .stride(1, 1)
//                        .padding(2, 2)
////                        .dist(new NormalDistribution(0, 1e-4))
//                        .nOut(64)
//                        .build())
//                .layer(4, new LocalResponseNormalization.Builder(4, 1e-3/9, 0.75).build())
//                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
//                        .name("pool2")
//                        .build())
//                .layer(6, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(384)
////                        .dist(new NormalDistribution(4e-3, 4e-2))
//                        .dropOut(0.5)
//                        .build())
//                .layer(7, new DenseLayer.Builder()
//                        .name("ffn2")
//                        .nOut(192)
////                        .dist(new NormalDistribution(4e-3, 4e-2))
//                        .dropOut(0.5)
//                        .build())
//                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(numLabels)
////                        .dist(new NormalDistribution(4e-3, 1/192.0))
//                        .activation("softmax")
//                        .build())

//http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf
//                .list(11)
//                .layer(0, new ConvolutionLayer.Builder(7, 7)
//                        .name("cnn1")
//                        .nIn(CHANNELS)
//                        .nOut(96)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
//                        .name("pool1")
//                        .build())
//                .layer(2, new LocalResponseNormalization.Builder().build())
//                .layer(3, new ConvolutionLayer.Builder(5, 5)
//                        .name("cnn2")
//                        .nOut(256)
//                        .build())
//                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
//                        .name("pool2")
//                        .build())
//                .layer(5, new LocalResponseNormalization.Builder().build())
//                .layer(6, new ConvolutionLayer.Builder(3, 3)
//                        .name("cnn3")
//                        .nOut(384)
//                        .build())
//                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
//                        .name("pool3")
//                        .build())
//                .layer(8, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(512)
//                        .dropOut(0.5)
//                        .build())
//                .layer(9, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(512)
//                        .dropOut(0.5)
//                        .build())
//                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(numLabels)
//                        .activation("softmax")
//                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(HEIGHT, WIDTH, CHANNELS);

        MultiLayerNetwork network = new MultiLayerNetwork(builder.build());
        network.init();

        // Listeners
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(listenerFreq).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq), paramListener));

        // Early Stopping

//        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
//        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
//                .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
//                .evaluateEveryNEpochs(1)
//                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
//                .scoreCalculator(new DataSetLossCalculator(mnistTest512, true))     //Calculate test set score
//                .modelSaver(saver)
//                .build();
//
//        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,configuration,mnistTrain1024);
//
//        //Conduct early stopping training:
//        EarlyStoppingResult result = trainer.fit();
//        System.out.println("Termination reason: " + result.getTerminationReason());
//        System.out.println("Termination details: " + result.getTerminationDetails());
//        System.out.println("Total epochs: " + result.getTotalEpochs());
//        System.out.println("Best epoch number: " + result.getBestModelEpoch());
//        System.out.println("Score at best epoch: " + result.getBestModelScore());
//
//        //Print score vs. epoch
//        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
//        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
//        Collections.sort(list);
//        System.out.println("Score vs. Epoch:");
//        for( Integer i : list){
//            System.out.println(i + "\t" + scoreVsEpoch.get(i));
//        }


        log.info("Train model....");
        // one epoch
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(trainIter);

        log.info("**********Last Score***********");
        log.info("{}", network.score());

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats());

        log.info("****************Example finished********************");

    }

    public static void main(String[] args) throws Exception {
        new MSRA_CFW().run(args);
    }

}