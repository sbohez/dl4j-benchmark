package org.deeplearning4j.train.misc;

import com.beust.jcommander.IStringConverter;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.train.DataLoadingMethod;
import org.deeplearning4j.train.config.CsvCompressionCodec;

/**
 * Created by Alex on 24/07/2016.
 */
public class EnumConverters {
    public static class RepartitionEnumConverter implements IStringConverter<Repartition>{
        @Override
        public Repartition convert(String s) {
            return Repartition.valueOf(s);
        }
    }

    public static class RepartitionStrategyEnumConverter implements IStringConverter<RepartitionStrategy>{
        @Override
        public RepartitionStrategy convert(String s) {
            return RepartitionStrategy.valueOf(s);
        }
    }

    public static class DataLoadingMethodEnumConverter implements IStringConverter<DataLoadingMethod>{
        @Override
        public DataLoadingMethod convert(String s) {
            return DataLoadingMethod.valueOf(s);
        }
    }

    public static class CsvCompressionCodecConverter implements IStringConverter<CsvCompressionCodec>{
        @Override
        public CsvCompressionCodec convert(String s) {
            return CsvCompressionCodec.valueOf(s);
        }
    }

    public static class RDDTrainingApproachConverter implements IStringConverter<RDDTrainingApproach>{
        @Override
        public RDDTrainingApproach convert(String s){
            return RDDTrainingApproach.valueOf(s);
        }
    }


}
