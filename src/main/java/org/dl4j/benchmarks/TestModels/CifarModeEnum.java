package org.dl4j.benchmarks.TestModels;

/**
 * ImageNet DataMode
 *
 * Defines which dataset between object recognition (CLS) and location identification (DET).
 * Also defines whether its train, cross validation or test phase
 */
public enum CifarModeEnum {
    BATCH_NORM, FULL_SIGMOID, QUICK;
}
