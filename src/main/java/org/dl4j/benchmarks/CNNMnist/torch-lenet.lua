-- Torch7 Lenet
--
-- Reference: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua

require 'torch'
require 'nn'
require 'optim'
require "cutorch"
require 'src/main/resources/torch-data/dataset-mnist'
require 'src/main/java/org/dl4j/benchmarks/Utils/benchmark-util'

total_time = sys.clock()
torch.manualSeed(42)
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-gpu', false, 'boolean flag to use gpu for training')
cmd:option('-cudnn', false, 'boolean flag to use cudnn for training')
config = cmd:parse(arg)

-- Lessons learned:
--    requires batch and numExamples to be divisable without remainder
--    harder to debug and research than python
--    More steps to apply gpu vs caffe and dl4j

opt = {
    gpu = config.gpu,
    usecuDNN = config.cudnn,
    max_epoch = 11,
    numExamples = 1000 , -- numExamples
    numTestExamples = 1000,
    batchSize = 100,
    testBatchSize = 100,
    noutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    ninputs = 28*28,
    coefL2 = 5e-4,
    nGPU = 1,
    learningRate = 1e-2,
    weightDecay = opt.coefL2,
    nesterov = true,
    momentum =  0.9,
    dampening = 0
}

if opt.gpu then print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name) end

optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    nesterov = opt.nesterov,
    momentum =  opt.momentum,
    dampening = opt.dampening
}

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {opt.height, opt.width}

------------------------------------------------------------
-- print('Load data')
data_load_time = sys.clock()
trainData = mnist.loadTrainSet(opt.numExamples, geometry)
trainData:normalizeGlobal()

testData = mnist.loadTestSet(opt.numTestExamples, geometry)
testData:normalizeGlobal()

data_load_time = sys.clock() - data_load_time

------------------------------------------------------------
--print('Build model')
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

if(opt.gpu) then
    require 'cunn'
    model:cuda()
    model = util.convertCuda(model, opt.usecuDNN, opt.nGPU)
end

local parameters,gradParameters = model:getParameters()
criterion = util.applyCuda(opt.gpu, nn.CrossEntropyCriterion())

------------------------------------------------------------
--print('Train model')

function train(dataset)
    model:training()
--    loops from 1 to full dataset size by batchsize
    for t = 1,opt.numExamples,opt.batchSize do
        -- create mini batch
--        local inputs = opt.gpu and torch.CudaTensor(opt.batchSize,opt.channels,opt.height,opt.width) or torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width)
----        local inputs = opt.gpu and torch.CudaTensor(opt.batchSize,opt.height,opt.width) or torch.Tensor(opt.batchSize,opt.height,opt.width)
--        local targets = opt.gpu and torch.CudaTensor(opt.batchSize):zero() or torch.zeros(opt.batchSize)
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
        inputs = opt.gpu and inputs:cuda() or inputs
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()
            -- get new parameters
            if x ~= parameters then parameters:copy(x) end
            -- reset gradients
            gradParameters:zero()
            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            local loss = criterion:forward(outputs, targets)
            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            return loss, gradParameters
        end
        optim.sgd(feval,parameters,optimState)
    end
end

------------------------------------------------------------
--print('Evaluate')

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

function test(dataset)
    print('Eval')
    model:evaluate()
    -- test over given dataset
    for t = 1,dataset:size(),opt.testBatchSize do
        -- disp progress
        xlua.progress(t, dataset:size())

        -- create mini batch
        local inputs = util.applyCuda(opt.gpu, torch.Tensor(opt.batchSize,opt.channels,opt.height,opt.width))
        local targets = util.applyCuda(opt.gpu, torch.zeros(opt.batchSize))
        local k = 1
        for i = t,math.min(t+opt.testBatchSize-1,opt.numTestExamples) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone():resize(opt.channels,opt.height,opt.width)
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        -- test samples
        local preds = model:forward(inputs)
        -- confusion:
        confusion:batchAdd(preds, targets)
    end
    -- print confusion matrix
    confusion:updateValids()
    print(confusion)
    print('Accuracy: ', confusion.totalValid * 100)
    confusion:zero()

end

train_time = sys.clock()
for _ = 1,opt.max_epoch do
    print(_)
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
