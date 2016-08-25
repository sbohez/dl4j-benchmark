package org.deeplearning4j.basic;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.BlockManager;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.train.functions.sequence.FromSequenceFilePairFunction;
import org.deeplearning4j.train.functions.sequence.ToSequenceFilePairFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 25/08/2016.
 */
public class TestOOM {

    public static void main(String[] args){

        int numDSObjects = 102400;
        String dataDir = "file:///C:/Temp/TestOOM/";


        SparkConf conf = new SparkConf();
        conf.setAppName("RunTrainingTests");
        conf.setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        List<Integer> intList = new ArrayList<>();
        for(int i = 0; i< numDSObjects; i++ ) intList.add(i);
        JavaRDD<Integer> intRDD = sc.parallelize(intList);

        //Export sequence file:
//        JavaRDD<DataSet> data3 = intRDD.map(new OtherGenerateDataFunction(new int[]{8,1000}));
        JavaRDD<DataSet> data3 = intRDD.map(new OtherGenerateDataFunction(new int[]{50,10000}));    //4MB total
        JavaPairRDD<Text,BytesWritable> pairRDD = data3.mapToPair(new ToSequenceFilePairFunction());
        pairRDD.saveAsHadoopFile(dataDir, Text.class, BytesWritable.class, SequenceFileOutputFormat.class);

        JavaPairRDD<Text,BytesWritable> sequenceFile = sc.sequenceFile(dataDir, Text.class, BytesWritable.class);
        JavaRDD<DataSet> trainData = sequenceFile.map(new FromSequenceFilePairFunction());

//        trainData.cache();
//        trainData.persist(StorageLevel.MEMORY_ONLY());
//        trainData.persist(StorageLevel.MEMORY_AND_DISK());
//        trainData.persist(StorageLevel.MEMORY_ONLY_SER());
        trainData.persist(StorageLevel.MEMORY_AND_DISK_SER());

        trainData.count();

//        ByteArrayOutputStream

    }

}
