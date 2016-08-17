-- Torch7 Lenet
--
-- Reference: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua

-- Lessons learned:
--    requires batch and numExamples to be divisable without remainder
--    harder to debug and research than python
--    More steps to apply gpu vs caffe and dl4j

require 'torch'
require 'nn'
require 'optim'
require 'logroll'
require "cutorch"
require 'dl4j-core-benchmark/src/main/resources/torch-data/dataset-mnist'
require 'dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/benchmark-util'

log = logroll.print_logger()
log.level = logroll.DEBUG

local cmd = torch.CmdLine()
cmd:option('-gpu', false, 'boolean flag to use gpu for training')
cmd:option('-cudnn', true, 'boolean flag to use cudnn for training')
cmd:option('-multi', false, 'boolean flag to use multi-gpu for training')
config = cmd:parse(arg)
if config.multi then print("Multi-GPU Not Implemented Yet") end

local opt = {
    gpu = config.gpu,
    usecuDNN = config.cudnn,
    max_epoch = 15,
    nExamples = 60000 ,
    nTestExamples = 10000,
    batchSize = 100,
    testBatchSize = 100,
    noutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    ninputs = 28*28,
    nGPU = 1,
    learningRate = 1e-2,
    weightDecay = 5e-4,
    nesterov = true,
    momentum =  0.9,
    dampening = 0,
    threads = 8,
    logger = log.level == logroll.DEBUG,
    plot = false,

}

local total_time = sys.clock()
torch.manualSeed(42)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.multi then opt.nGPU = 4 end

if opt.gpu then print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name) end

local optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    nesterov = opt.nesterov,
    momentum =  opt.momentum,
    dampening = opt.dampening
}

local classes = {'1','2','3','4','5','6','7','8','9','10'}
local geometry = {opt.height, opt.width}
confusion = optim.ConfusionMatrix(classes)

------------------------------------------------------------
log.debug('Load data...')

data_load_time = sys.clock()
trainData, testData =  util.loadData(opt.nExamples, opt.nTestExamples, geometry)
data_load_time = sys.clock() - data_load_time

------------------------------------------------------------
log.debug('Build model...')

model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(1, 20, 5, 5))
model:add(nn.Identity())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(20, 50, 5, 5))
model:add(nn.Identity())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(50*4*4))
model:add(nn.Linear(50*4*4, 500))
model:add(nn.ReLU(true))
model:add(nn.Linear(500, #classes))
model = util.updateParams(model)

if(opt.gpu) then model = util.convertCuda(model, opt.usecuDNN, opt.nGPU) end

local parameters,gradParameters = model:getParameters()
criterion = util.applyCuda(opt.gpu, nn.CrossEntropyCriterion())

------------------------------------------------------------
--print('Train model')

function train(dataset)
    log.debug('Train model...')
    model:training()
--    loops from 1 to full dataset size by batchsize
    for t = 1,dataset.size(), opt.batchSize do
        -- disp moving progress for data load
        if opt.logger then xlua.progress(t, dataset:size()) end
        -- create mini batch
        local inputs = util.applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = util.applyCuda(opt.gpu, torch.zeros(opt.batchSize))
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone():resize(opt.channels,opt.height,opt.width)
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            collectgarbage()
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()
            local outputs = model:forward(inputs)
            local loss = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            -- update confusion
            return loss, gradParameters
        end
        optim.sgd(feval,parameters,optimState)
    end
    confusion:updateValids()
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100 }
    -- plot errors
    if opt.plot then
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        trainLogger:plot()
    end
    confusion:zero()
    if opt.logger then print(confusion) end
end

------------------------------------------------------------
--Evaluate model

function test(dataset)
    log.debug('Evaluate model...')
    model:evaluate()
    for t = 1,dataset:size(),opt.testBatchSize do
        -- create mini batch
        local inputs = util.applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = util.applyCuda(opt.gpu, torch.zeros(opt.batchSize))
        local k = 1
        for i = t,math.min(t+opt.testBatchSize-1,opt.nTestExamples) do
            -- disp moving progress for data load
            if opt.logger then xlua.progress(t, dataset:size()) end
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone():resize(opt.channels,opt.height,opt.width)
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        local preds = model:forward(inputs)
        confusion:batchAdd(preds, targets)
        -- plot errors
        if opt.plot then
            testLogger:style{['% mean class accuracy (test set)'] = '-'}
            testLogger:plot()
        end
    end
    -- print confusion matrix
    confusion:updateValids()
    if opt.logger then print(confusion) end
    print('Accuracy: ', confusion.totalValid * 100)
end

train_time = sys.clock()
for epoch = 1,opt.max_epoch do
    log.debug('<trainer> on training set:')
    log.debug("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    train(trainData)
end
train_time = sys.clock() - train_time

test_time = sys.clock()
test(testData)
test_time = sys.clock() - test_time
total_time = sys.clock() - total_time

print("****************Example finished********************")
util.printTime('Data load', data_load_time)
util.printTime('Train', train_time)
util.printTime('Test', test_time)
util.printTime('Total', total_time)
