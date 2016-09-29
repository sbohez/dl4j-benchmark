package org.deeplearning4j.Utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Utils for all models
 */
public class DL4J_Utils {
    private static Logger log = LoggerFactory.getLogger(DL4J_Utils.class);
    public final static int buffer = 24;
    public final static int avgFrequency = 3;

    public static void printTime(String name, long ms){
        log.info(name + " time: {} min, {} sec | {} milliseconds",
                TimeUnit.MILLISECONDS.toMinutes(ms),
                TimeUnit.MILLISECONDS.toSeconds(ms) -
                        TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(ms)),
                ms);
    }

    public static ParallelWrapper multiGPUModel(MultiLayerNetwork network, int buffer, int workers, int avgFrequency) {
        return new ParallelWrapper.Builder(network)
                .prefetchBuffer(buffer)
                .workers(workers)
                .averagingFrequency(avgFrequency)
                .build();
    }

    public static void train(MultiLayerNetwork network, DataSetIterator data, int numGPUs){
//        org.nd4j.jita.conf.CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true)
//                .setMaximumDeviceCache(4L * 1024L * 1024L).setMaximumHostCache(8L * 1024L * 1024L)
//                .setMaximumGridSize(512).setMaximumBlockSize(512).allowCrossDeviceAccess(true);

        if(numGPUs > 0) {
            ParallelWrapper wrapper = multiGPUModel(network, buffer, numGPUs, avgFrequency);
            wrapper.fit(data);
        } else {
            network.fit(data);
        }

    }

}
