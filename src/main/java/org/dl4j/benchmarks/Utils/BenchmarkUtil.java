package org.dl4j.benchmarks.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 */
public class BenchmarkUtil {
    private static Logger log = LoggerFactory.getLogger(BenchmarkUtil.class);

    public static void printTime(String name, long ms){
        log.info(name + " time: %d min, %d sec | %d milliseconds",
                TimeUnit.MILLISECONDS.toMinutes(ms),
                TimeUnit.MILLISECONDS.toSeconds(ms) -
                        TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(ms)),
                ms);

    }

}
