package org.deeplearning4j.basic;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Alex on 25/08/2016.
 */
@AllArgsConstructor
public class OtherGenerateDataFunction implements Function<Integer,DataSet> {

    private final int[] shape;

    @Override
    public DataSet call(Integer v1) throws Exception {
        INDArray f = Nd4j.zeros(shape);
        INDArray l = Nd4j.zeros(shape);

        return new DataSet(f,l);
    }
}
