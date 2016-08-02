import time
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import Utils.benchmark_util as util

util.printTime("Stuff", time.time() - time.time())
print util.NUM_GPUS