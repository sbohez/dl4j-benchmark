package org.deeplearning4j.basic;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;

/**
 * Created by Alex on 24/08/2016.
 */
public class DebugTest {

    public static void main(String[] args) throws Exception {

//        INDArray first = Nd4j.create(8,1000);
//        INDArray second = Nd4j.create(8, 1000);

        INDArray first = Nd4j.create(8, 10000);
        INDArray second = Nd4j.create(8, 10000);

        DataSet ds = new DataSet(first, second);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);

        oos.writeObject(ds);
        oos.close();

        byte[] bytes = baos.toByteArray();

        System.out.println("Total bytes: " + bytes.length);


    }

}
